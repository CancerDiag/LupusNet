import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from models.resnet_custom import resnet50_new
from monai.networks.nets import ViT


class Gated_Attention(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Gated_Attention, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class LupusNet(nn.Module):
    def __init__(self,path_input_dim=2048, gate = True, size_arg = "small", dropout = False, k_sample=1, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(LupusNet, self).__init__()
        self.size_dict = {"small": [path_input_dim, 512, 256], "big": [path_input_dim, 512, 384]}
        size = self.size_dict[size_arg]
        if path_input_dim == 384:
            size = [384, 384, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = Gated_Attention(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1]*2, n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        hidden_dim = 512
        num_layers = 1
        bdirect = True
        mul_fact = 2
        self.lstm = nn.LSTM(2048, hidden_dim, num_layers, bidirectional=bdirect)
        self.linear1 = nn.Linear(hidden_dim*mul_fact, 512)
        self.multihead_attn = nn.MultiheadAttention(path_input_dim, 16)

        self.resnet = resnet50_new(pretrained=True)

        ## freeze the resnet except the last block
        for param in self.resnet.parameters():
            param.requires_grad = False
        # for param in self.resnet.layer4.parameters():
        #     param.requires_grad = True
            

        # self.vit_net = ViT(in_channels=3, img_size=(224,224), dropout_rate =0.3, num_layers  = 1, mlp_dim = 512, patch_size = 16, pos_embed_type='learnable', classification=False, spatial_dims=2, hidden_size = path_input_dim, num_heads=2)
        # self.linear1 = nn.Linear(path_input_dim, 512)
        # self.bias = nn.Parameter(torch.zeros(3,dtype=torch.float,device="cpu"))

        self.s = nn.Parameter(torch.zeros(3,dtype=torch.float,device="cpu"))
        # self.s1 = nn.Parameter(torch.zeros(1,dtype=torch.float,device="cpu"))
        # self.y = nn.Parameter(torch.ones(1,dtype=torch.float,device="cpu"))

        # self.classifiers1 = nn.Linear(size[1], n_classes)
        # self.classifiers2 = nn.Linear(size[1], n_classes)

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        self.lstm = self.lstm.to(device)
        self.multihead_attn = self.multihead_attn.to(device)
        # self.vit_net = self.vit_net.to(device)
        self.linear1 = self.linear1.to(device)
        # self.classifiers1 = self.classifiers1.to(device)
        # self.classifiers2 = self.classifiers2.to(device)



    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = self.resnet(h)  ## extract glom features N*d
        device = h.device
        h_copy = h
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A

        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h)

        h_copy, attn_w = self.multihead_attn(h_copy, h_copy, h_copy)
        x_lstm, _ =  self.lstm(h_copy)
        x_lstm = x_lstm[-1, :].unsqueeze(0)
        x_lstm = self.linear1(x_lstm)

        # x_vit = self.vit_net(h_copy.unsqueeze(0))[:, 0]
        # x_vit = self.linear1(x_vit)
        # logits_l = self.classifiers1(M)
        # logits_r = self.classifiers2(x_vit)
        s = F.softmax(self.s[:-1], dim = 0)
        ## print devices of s0, s1, y, M, x_lstm
        # print(s0.device, s1.device, self.y.device, M.device, x_lstm.device)
        M = s[2]*(s[0]*M + s[1]*x_lstm)
        # M = torch.cat((M, x_lstm), dim=1)
        # A_m, M = self.weighted_attn(M) ## 2x1
        # A_m = torch.transpose(A_m, 1, 0)     ## 1x2
        # A_m = F.softmax(A_m, dim=1)  # softmax over N
        # M = torch.mm(A_m, M)


        logits_m = self.classifiers(M)
        
        Y_hat = torch.topk(logits_m, 1, dim = 1)[1]
        Y_prob = F.softmax(logits_m, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        # return (logits_l, logits_m, logits_r), Y_prob, Y_hat, A_raw, results_dict
        return logits_m, Y_prob, Y_hat, (A_raw, attn_w), results_dict
        # return logits_m, Y_prob, Y_hat, A_raw, results_dict