import argparse
import torch

#Method that is used to create the training settings for the Planetoid and Amazon Dataset
def Get_Train_Settings():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
    #parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=30, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-16,help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
    parser.add_argument('--layers',type=int,default=2, help='Number of Layers')
    parser.add_argument('--device',type=str,default='cpu',help='Device used to train the model')
    parser.add_argument('--x_dim',type=int,default=3,help='Dimension of the coordinates of individual nodes')
    parser.add_argument('--train_set_size',type=int,default=20,help='Number of training samples from each class')
    parser.add_argument('--val_set_size',type=int,default=30,help='Number of validation samples from each class')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


#Method that is used to create the training settings for the QM9 Dataset
def Get_Train_Settings_QM9():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-16,help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=3,help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
    parser.add_argument('--layers',type=int,default=5, help='Number of Layers')
    parser.add_argument('--device',type=str,default='cuda',help='Device used to train the model')
    parser.add_argument('--train_set_size',type=int,default=100000,help='Number of training samples from each class')
    parser.add_argument('--test_pct',type=float,default=0.1,help='Percentage of the total samples used for test')
    parser.add_argument('--batch_size',type=int,default=96,help='batch size')
    parser.add_argument('--property',type=int,default=4,help='Target Property Index')
    parser.add_argument('--input_dim',type=int,default=11,help='Input Dimension Of The Feature Embedding')
    parser.add_argument('--flag',type=str,default='reg',help='Flag for prediction operations')
    parser.add_argument('--log_interval',type=int,default=20,help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval',type=int,default=1,help='how many epochs to wait before logging test')
    parser.add_argument('--readout',type=str,default='add',help='readout function')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


#Method that is used to create the training settings for the OGB Dataset
def Get_Train_Settings_OGB():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=777, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=401,help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,help='Learning rate.')
    parser.add_argument('--decay_rate', type=float, default=0.7,help='decay_rate')
    parser.add_argument('--hidden', type=int, default=64,help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
    parser.add_argument('--layers',type=int,default=3, help='Number of Layers')
    parser.add_argument('--device',type=str,default='cuda',help='Device used to train the model')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--dataset_name',type=str,default="ogbg-molhiv",help='dataset_name')
    parser.add_argument('--input_dim',type=int,default=9,help='Input Dimension Of The Feature Embedding')
    parser.add_argument('--flag',type=str,default='cls',help='Flag for prediction operations')
    parser.add_argument('--step_size',type=int,default=5,help='Step Size')
    parser.add_argument('--task_type',type=str,default="classification",help='Dataset Task Type')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

#Method that is used to get the training settings for the ZINC dataset
def Get_Train_Settings_ZINC():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=41, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,help='Number of epochs to train.')
    parser.add_argument('--init_lr',type=float,default=0.0007,help='initial learning rate')
    parser.add_argument('--lr_reduce_factor',type=float,default=0.5,help='initial learning rate')
    parser.add_argument('--lr_schedule_patience',type=int,default=15,help='learning rate schedule patience')
    parser.add_argument('--min_lr',type=float,default=1e-6,help='min_lr')
    parser.add_argument('--weight_decay',type=float,default=0.0,help='weight_decay')
    parser.add_argument('--print_epoch_interval',type=int,default=5,help='print_epoch_interval')
    parser.add_argument('--max_time',type=int,default=24,help='max_time')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size')
    parser.add_argument('--dataset_name',type=str,default="ZINC",help='dataset_name')
    parser.add_argument('--device',type=str,default='cuda',help='Device used to train the model')
    parser.add_argument('--flag',type=str,default='reg',help='Flag for prediction operations')
    parser.add_argument('--hidden', type=int, default=64,help='Number of hidden units.')
    parser.add_argument('--layers',type=int,default=5, help='Number of Layers')
    parser.add_argument('--input_dim',type=int,default=9,help='Input Dimension Of The Feature Embedding')
    parser.add_argument('--readout',type=str,default='mean',help='readout function')
    parser.add_argument('--dropout', type=float, default=0.0,help='Dropout rate (1 - keep probability).')
    parser.add_argument('--in_feat_dropout', type=float, default=0.0,help='Input Feature Dropout Rate')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args