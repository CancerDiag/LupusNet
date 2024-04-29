from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_val_class_acc = []
    all_test_class_acc = []

    all_val_f1 = []
    all_test_f1 = []
    all_val_class_f1 = []
    all_test_class_f1 = []


    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)

        val_classes_accs = {}
        test_classes_accs = {}
        val_classes_f1s = {}
        test_classes_f1s = {}

        fold_result  = train(datasets, i, args)

        for n in range(args.n_classes):
            val_classes_accs['val_class_{}_acc'.format(n)] = fold_result['val_class_wise_acc'][n]
            test_classes_accs['test_class_{}_acc'.format(n)] = fold_result['test_class_wise_acc'][n]
            val_classes_f1s['val_class_{}_f1'.format(n)] = fold_result['val_class_wise_f1'][n]
            test_classes_f1s['test_class_{}_f1'.format(n)] = fold_result['test_class_wise_f1'][n]


        results = fold_result['results_dict']
        all_test_auc.append(fold_result['test_auc'])
        all_val_auc.append(fold_result['val_auc'])
        all_test_acc.append(fold_result['test_acc'])
        all_val_acc.append(fold_result['val_acc'])
        all_val_f1.append(fold_result['val_f1_score'])
        all_test_f1.append(fold_result['test_f1_score'])        
        
        all_val_class_acc.append(val_classes_accs)
        all_test_class_acc.append(test_classes_accs)
        all_val_class_f1.append(val_classes_f1s)
        all_test_class_f1.append(test_classes_f1s)


        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df1 = pd.DataFrame({'Result_dir': args.results_dir + '/' +args.exp_code , 'folds': folds, 'val_auc': all_val_auc})
    final_df2 = pd.DataFrame(all_val_class_acc)
    space_df = pd.DataFrame({'  ':['']})
    final_df3 = pd.DataFrame({'test_auc': all_test_auc})
    final_df4 = pd.DataFrame(all_test_class_acc)
    final_df5 = pd.DataFrame(all_val_class_f1)
    final_df6 = pd.DataFrame({'val_f1' : all_val_f1})
    final_df7 = pd.DataFrame(all_test_class_f1)
    final_df8 = pd.DataFrame({'test_f1' : all_test_f1})
    final_df9 = pd.DataFrame({'val_acc' : all_val_acc, 'test_acc': all_test_acc})

    final_df = pd.concat([final_df1, final_df2, space_df, final_df3, final_df4, space_df, final_df5, final_df6, space_df, 
                          final_df7, final_df8, space_df, space_df, final_df9], axis=1)

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary_'+args.exp_code + '.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.00001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--labels_list', type=str, nargs= '+', default=None, help='Name of classes')

parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=True, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=1, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--path_input_dim', type=int, default=1024, help='Size of patch embedding size (384 for DINO, 1024 for resnet)')
parser.add_argument('--dataset_csv_path', type=str, default=None, help='dataset csv path for subtyping')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.labels_list is None:
    raise NotImplementedError
else:
    args.n_classes = len(args.labels_list)


if args.n_classes is not None:
    label_dictionary = {}
    for i in range(args.n_classes):
        label_dictionary['subtype_{}'.format(i+1)] = i
else:
    raise NotImplementedError


dataset = Generic_MIL_Dataset(csv_path = args.dataset_csv_path,
                        data_dir= args.data_root_dir,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = label_dictionary,
                        patient_strat=False,
                        ignore=[])

if args.model_type in ['clam_sb', 'clam_mb']:
    assert args.subtyping
        
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is not None:
    args.split_dir = os.path.join('splits', args.split_dir)
else:
    raise NotImplementedError

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")