import time
import numpy as np
import torch
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.lupusnet import LupusNet
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc, f1_score
from sklearn.metrics import classification_report, confusion_matrix


class F1_Logger(object):
    """F1 score logger"""
    def __init__(self):
        super(F1_Logger, self).__init__()
        self.initialize()

    def initialize(self):
        self.y = []
        self.y_hat = []
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.y.append(Y)
        self.y_hat.append(Y_hat)
    
    def get_summary(self):
        y_true = self.y 
        y_pred = self.y_hat
        if len(y_pred) == 0: 
            f1 = None
        else:
            f1_per_class = f1_score(y_true, y_pred, average=None)
            f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return f1_per_class, f1_macro
    


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {'path_input_dim': args.path_input_dim,"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.B > 0:
        model_dict.update({'k_sample': args.B})
    
    if args.inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()

    model = LupusNet(**model_dict, instance_loss_fn=instance_loss_fn)

    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        start_time_epoch = time.time()
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
        stop = validation(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break
        print('Epoch {} took {:.2f} sec'.format(epoch, time.time() - start_time_epoch))

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, val_acc_logger, val_f1_score, val_f1_class_wise = summary(model, val_loader, args, cur, validation = True)
    print('Val error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(val_error, val_auc, val_f1_score))

    results_dict, test_error, test_auc, acc_logger, test_f1_score, test_f1_class_wise = summary(model, test_loader, args, cur)
    print('Test error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(test_error, test_auc, test_f1_score))

    val_class_wise_acc = []
    test_class_wise_acc = []

    for i in range(args.n_classes):
        val_acc, val_correct, val_count = val_acc_logger.get_summary(i)
        val_class_wise_acc.append(val_acc)

        acc, correct, count = acc_logger.get_summary(i)
        test_class_wise_acc.append(acc)

        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    # output = results_dict, test_auc, val_auc, 1-test_error, 1-val_error, val_class_wise_acc, test_class_wise_acc, val_f1_score, test_f1_score, val_f1_class_wise, test_f1_class_wise

    output_dict = {'results_dict': results_dict, 'test_auc': test_auc, 'val_auc': val_auc, 'test_acc': 1-test_error, 
                   'val_acc': 1-val_error, 'val_class_wise_acc': val_class_wise_acc, 'test_class_wise_acc': test_class_wise_acc, 
                   'val_f1_score': val_f1_score, 'test_f1_score': test_f1_score, 'val_class_wise_f1': val_f1_class_wise, 
                   'test_class_wise_f1': test_f1_class_wise}

    return output_dict



def train_loop(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    f1_logger = F1_Logger()
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=False)
        # logits_l, logits_m, logits_r = logits
        logits_m = logits

        acc_logger.log(Y_hat, label)
        f1_logger.log(Y_hat, label)
        
        # loss_l = loss_fn(logits_l, label)
        loss_m = loss_fn(logits_m, label)
        # loss_r = loss_fn(logits_r, label)

        # loss_value = loss_l.item() + loss_m.item() + loss_r.item()
        loss_value = loss_m.item() 


        # instance_loss = instance_dict['instance_loss']
        # inst_count+=1
        # instance_loss_value = instance_loss.item()
        # train_inst_loss += instance_loss_value
        
        # total_loss = bag_weight * loss_m + (1-bag_weight) * instance_loss   ##0.7
        # total_loss = 0.9 * (loss_l + loss_m + loss_r) + (1-0.9) * instance_loss
        total_loss = 0.8 * (loss_m) + (1-0.8) * instance_loss
        # total_loss = loss_m


        # inst_preds = instance_dict['inst_preds']
        # inst_labels = instance_dict['inst_labels']
        # inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, 0, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    
    f1_scores_class_wise, f1_macro = f1_logger.get_summary()

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
        
    for i in range(n_classes):
        print('class {}: f1 {}'.format(i, f1_scores_class_wise[i]))

        if writer and f1_scores_class_wise is not None:
            writer.add_scalar('train/class_{}_f1'.format(i), f1_scores_class_wise[i], epoch)

   
    print('f1_macro: {}'.format(f1_macro))
    
    if writer:
        writer.add_scalar('train/f1_macro', f1_macro, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def validation(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    f1_logger = F1_Logger()
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=False)

            # logits_l, logits_m, logits_r = logits
            logits_m = logits

            acc_logger.log(Y_hat, label)
            f1_logger.log(Y_hat, label)

            # loss_l = loss_fn(logits_l, label)
            loss_m = loss_fn(logits_m, label)
            # loss_r = loss_fn(logits_r, label)

            # loss_value = loss_l.item() + loss_m.item() + loss_r.item()
            loss_value = loss_m.item() 
            val_loss += loss_value

            # instance_loss = instance_dict['instance_loss']
            
            # inst_count+=1
            # instance_loss_value = instance_loss.item()
            # val_inst_loss += instance_loss_value

            # inst_preds = instance_dict['inst_preds']
            # inst_labels = instance_dict['inst_labels']
            # inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    
    f1_scores_class_wise, f1_macro = f1_logger.get_summary()
    
    if writer:
        writer.add_scalar('val/f1_macro', f1_macro, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    for i in range(n_classes):
        print('class {}: f1 {}'.format(i, f1_scores_class_wise[i]))
        
        if writer and f1_scores_class_wise is not None:
            writer.add_scalar('val/class_{}_f1'.format(i), f1_scores_class_wise[i], epoch)

    print('f1_macro: {}'.format(f1_macro))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, args, current_fold, validation = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    y_pred = []
    y_true = []

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        y_pred.append(Y_hat.item())
        y_true.append(label.item())
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    
    f1_score = calculate_macro_f1(y_pred, y_true)
    f1_score_class_wise = calculate_class_wise_f1(y_pred, y_true)
    cm = confusion_matrix(y_true, y_pred)
    cm_array_df = pd.DataFrame(cm, index=args.labels_list, columns=args.labels_list)

    ## Confusion Matrix
    results_dir_cm = os.path.join(args.results_dir, 'confusion_matrix')
    if not os.path.isdir(results_dir_cm):
        os.mkdir(results_dir_cm)

    sn.set(font_scale=0.8) # for label size
    sn.heatmap(cm_array_df, annot=True, annot_kws={"size": 12})
    plt.title('Confusion Matrix'+ args.exp_code)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if validation:
        plt.savefig(results_dir_cm+ '/confusion_matrix_val_{}.png'.format(current_fold))
    else:
        plt.savefig(results_dir_cm+ '/confusion_matrix_test_{}.png'.format(current_fold))

    plt.close()

    if args.n_classes == 2:
        print('2 classes : ', all_labels.shape, all_probs.shape)
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        print('3 classes : ', all_labels.shape, all_probs.shape)
        binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        for class_idx in range(args.n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger, f1_score, f1_score_class_wise
