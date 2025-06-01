import torch
import torch.optim as optim
import torch.nn as nn
from Utils import Get_sparse_adjacency_matrix
from Params import Get_Train_Settings_OGB
from model_v2 import Custom_GNN
from Load_Dataset import load_OGBG_Dataset
from torch_geometric.nn import global_mean_pool
from ogb.graphproppred import Evaluator
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Get the training related settings for the OGB Dataset
args=Get_Train_Settings_OGB()

#Set the Criterion for both Classification and Regression tasks
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


#Method used to train the model
def train(model, device, loader, optimizer, task_type):
    model.train()
    loss_all = 0
    
    for data,label in loader:
        
        src,dest=data.edges()   #Extract the src and est nodes of the edges
        
        indices=torch.tensor([src.tolist(),dest.tolist()]) #Indices tensor
        
        input=data.ndata['feat'].to(device).float()
        
        num_nodes=input.shape[0] #Extract the number of nodes
        
        batch_num_nodes=data.batch_num_nodes() #Get the number of nodes in current batch
        
        batch_size=len(batch_num_nodes) #Number of mini-batches
        
        node_batch_indices = torch.cat([
        
        torch.full((n,), i) for i, n in enumerate(batch_num_nodes)       #Sequentially assign nodes to batches
        ]).to(device)    
        
        A=Get_sparse_adjacency_matrix(indices,num_nodes,device) #Get the sparse adjacency matrix
        
        x=torch.ones(input.shape[0],args.hidden,device=device) #Initialize the coordinates
        
        pred = model(input,A,x)  #Get the prediction from the model   
        
        graph_level_logits=global_mean_pool(pred,node_batch_indices) #Node pooling for graph level prediction 
        

        optimizer.zero_grad()
        
        labels=label.to(device)
        if "classification" in task_type:
            loss = cls_criterion(graph_level_logits.to(torch.float32),labels.to(torch.float32))
        else:
            loss = reg_criterion(graph_level_logits.to(torch.float32), labels.to(torch.float32))
        
        loss.backward()
        
        loss_all += loss.item()
        
        optimizer.step()
    
    return loss_all / len(loader)


#Method used to evaluate the model
def eval(model, device, loader, evaluator):
    model.eval()
    
    y_true = []
    y_pred = []

    for data,label in loader:
        input=data.ndata['feat'].to(device).float() #Get the feature tensor
        
        src,dest=data.edges()   #Extract the src and est nodes of the edges

        indices=torch.tensor([src.tolist(),dest.tolist()]) #Indices tensor

        num_nodes=input.shape[0] #Extract the number of nodes

        A=Get_sparse_adjacency_matrix(indices,num_nodes,device) #Get the sparse adjacency matrix

        x=torch.ones(input.shape[0],args.hidden,device=device) #Initialize the coordinates
        
        labels=label.to(device)

        batch_num_nodes=data.batch_num_nodes() #Get the number of nodes in current batch

        node_batch_indices = torch.cat([
            torch.full((n,), i) for i, n in enumerate(batch_num_nodes)       #Sequentially assign nodes to batches
            ]).to(device)
        
        with torch.no_grad():
            pred = model(input,A,x)  #Get the prediction from the model
            graph_level_logits=global_mean_pool(pred,node_batch_indices) #Node pooling for graph level prediction    
        

        y_true.append(labels.detach().cpu())
        y_pred.append(graph_level_logits.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    
    return evaluator.eval(input_dict)


def main():
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset_name)


    #Dataloading
    train_loader,valid_loader,test_loader,num_classes,eval_metric=load_OGBG_Dataset(args.dataset_name,args.batch_size)
    
    model=Custom_GNN(args.layers,args.input_dim,args.hidden,1,args.dropout,args.device,args.flag)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,gamma=args.decay_rate)

    valid_curve = []
    test_curve = []
    train_curve = []
    trainL_curve = []

    for epoch in range(1, args.epochs + 1):
        print("Epoch {} training...".format(epoch))
        
        train_loss = train(model, args.device, train_loader, optimizer, args.task_type)

        scheduler.step()

        print('Evaluating...')

        train_perf = eval(model, args.device, train_loader, evaluator)
        valid_perf = eval(model, args.device, valid_loader, evaluator)
        test_perf =  eval(model, args.device, test_loader, evaluator)

        print('Train:', train_perf[eval_metric],
              'Validation:', valid_perf[eval_metric],
              'Test:', test_perf[eval_metric],
              'Train loss:', train_loss)
        
        train_curve.append(train_perf[eval_metric])
        valid_curve.append(valid_perf[eval_metric])
        test_curve.append(test_perf[eval_metric])
        trainL_curve.append(train_loss)


    if 'classification'==args.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)

    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)
    print('Finished test: {}, Validation: {}, epoch: {}, best train: {}, best loss: {}'
      .format(test_curve[best_val_epoch], valid_curve[best_val_epoch],
              best_val_epoch, best_train, min(trainL_curve)))
    
    with open("Exec_Results2.txt",'a') as f:
        f.write('Finished test: {}, Validation: {}, epoch: {}, best train: {}, best loss: {}'.format(test_curve[best_val_epoch], 
                                                                                                     valid_curve[best_val_epoch],
                                                                                                     best_val_epoch, best_train, 
                                                                                                  min(trainL_curve)))
    #Plot the performance
    fig,axis=plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    
    df=pd.DataFrame({
    'Epoch':np.array([e for e in range(1, args.epochs + 1)]),
    'Train Performance':np.array(train_curve),
    'Validation Performance':np.array(valid_curve),
    'Test Performance':np.array(test_curve),
    'Train Loss Curve':np.array(trainL_curve)
    })
    
    sns.lineplot(data=df,x='Epoch',y='Train Performance',ax=axis[0][0],label="Custom GNN")
    sns.lineplot(data=df,x='Epoch',y='Validation Performance',ax=axis[0][1],label='Custom GNN')
    sns.lineplot(data=df,x='Epoch',y='Test Performance',ax=axis[1][0],label='Custom GNN')
    sns.lineplot(data=df,x='Epoch',y='Train Loss Curve',ax=axis[1][1],label='Custom GNN')

    # Customizing the plot
    axis[0][0].set_title(args.dataset_name)
    axis[0][0].set_xlabel("Epoch")
    axis[0][0].set_ylabel("Training Performance")
    axis[0][0].tick_params(axis='x', rotation=45)
    axis[0][0].legend()
    axis[0][0].grid(True, linestyle='--', alpha=0.6)
    
    axis[0][1].set_title(args.dataset_name)
    axis[0][1].set_xlabel("Epoch")
    axis[0][1].set_ylabel("Validation Performance")
    axis[0][1].tick_params(axis='x', rotation=45)
    axis[0][1].legend()
    axis[0][1].grid(True, linestyle='--', alpha=0.6)

    axis[1][0].set_title(args.dataset_name)
    axis[1][0].set_xlabel("Epoch")
    axis[1][0].set_ylabel("Test Performance")
    axis[1][0].tick_params(axis='x', rotation=45)
    axis[1][0].legend()
    axis[1][0].grid(True, linestyle='--', alpha=0.6)

    axis[1][1].set_title(args.dataset_name)
    axis[1][1].set_xlabel("Epoch")
    axis[1][1].set_ylabel("Train Loss")
    axis[1][1].tick_params(axis='x', rotation=45)
    axis[1][1].legend()
    axis[1][1].grid(True, linestyle='--', alpha=0.6)

    # Show plot
    plt.tight_layout()
    plt.savefig(f"/home/mlrl/Suranjan/Custom_GNN/plots/OGBG/OGBG_ppa/OGBG Performance.png")
    plt.close()


#Trigger the main method
if __name__=="__main__":
    main()