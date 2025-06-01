#All the necessary Imports
import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import Adam
from Load_Dataset import load_QM9_Dataset
from Params import Get_Train_Settings_QM9
from Utils import compute_mean_mad
from model_v2 import Custom_GNN
import matplotlib.pyplot as plt
#from test import Custom_GNN


#Get the training related settings for the QM9 Dataset
args=Get_Train_Settings_QM9()


#Set the seed for the model
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#Load the QM9 dataset
train_loader,validation_loader,test_loader=load_QM9_Dataset(args.device,args.train_set_size,args.test_pct,args.batch_size)

# compute mean and mean absolute deviation of the target property
mean, mad = compute_mean_mad(train_loader, args.property)

#Initialize The Model
model = Custom_GNN(L=args.layers,input_dim=args.input_dim,hidden_dim=args.hidden,out_dim=args.hidden,dropout=args.dropout,device=args.device,flag=args.flag,readout=args.readout)

#print(model)


optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

loss_l1 = nn.L1Loss()



def train(epoch, loader, partition='train'):
    lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        
        input=data.x.to(args.device,dtype=torch.float32)
        A=torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.shape[1]),(data.x.shape[0],data.x.shape[0]),device=args.device,dtype=torch.float32)
        x=data.pos.to(args.device,dtype=torch.float32)
        batch=data.batch.to(args.device)
        label=data.y[:,args.property].to(args.device,dtype=torch.float32) 
        pred=model(input,A,x,batch).to(args.device)
        


        if partition == 'train':
            loss = loss_l1(pred, (label - mean) / mad)
            loss.backward()
            
            '''for name, param in model.named_parameters():
                if param.grad is not None:  # Check if the gradient exists
                    if torch.isnan(param.grad).any():  # Check for NaNs in the gradient
                        print(f"NaN detected in gradient of {name}")
                    else:
                        print(f"Gradient of {name} is valid")
                else:
                    print(f"No gradient found for {name}")'''


            for param in model.parameters():
                if (param.grad is not None) and (torch.isnan(param.grad).any()):
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
            
            optimizer.step()
        else:
            loss = loss_l1(mad * pred + mean, label)
        
        res['loss'] += loss.item() * args.batch_size
        res['counter'] += args.batch_size
        res['loss_arr'].append(loss.item())
        
        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
        
        if(i==2):
            break
    return res['loss'] / res['counter']


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    
    for epoch in range(0, args.epochs):
        train(epoch, train_loader, partition='train')
        if epoch % args.test_interval == 0:
            val_loss = train(epoch, validation_loader, partition='valid')
            test_loss = train(epoch, test_loader, partition='test')
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
        
        with open("Exec_Results.txt", "a") as f:
            f.write(f"Property:{args.property} Best: val loss: {res['best_val']:.4f}  test loss:{res['best_test']:.4f} epoch {res['best_epoch']}\n")

    #The plots
    fig=plt.figure(figsize=(10,10))
    plt.plot([(i+1) for i in range(args.epochs)],res['losess'])
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.title(f"QM9-{args.property}")
    plt.tight_layout()
    plt.savefig(f"/home/mlrl/Suranjan/Custom_GNN/plots/QM9_2/QM9-{args.property}.png")
    plt.close()