import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time
from tqdm import tqdm
import numpy as np
from Params import Get_Train_Settings_ZINC
from Utils import Get_sparse_adjacency_matrix
from Load_Dataset import load_ZINC_Dataset
#from model_v2 import Custom_GNN
from model_v3 import Custom_GNN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#Get the training related settings for the OGB Dataset
args=Get_Train_Settings_ZINC()

#Method used to compute the means absolute error 
def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE

#Method used to compute the L1 Loss
def get_loss(scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss

#Train method
def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0

    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device,dtype=torch.float32)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device,dtype=torch.float32)
        batch_targets = batch_targets.squeeze(-1).to(device,dtype=torch.float32)
        optimizer.zero_grad()
        
        src,dest=batch_graphs.edges() #Get the src and destination indices

        indices=torch.stack([src, dest]) #edge indices for Adjacency matrix

        num_nodes=batch_graphs.num_nodes() #No of nodes in the batch

        A=Get_sparse_adjacency_matrix(indices,num_nodes,device)
        
        x=torch.rand(num_nodes,args.hidden,device=device,dtype=torch.float32) #Initialize the coordinates
        
        batch_num_nodes = batch_graphs.batch_num_nodes()  # Returns a tensor of node counts per graph
        
        batch_indices = torch.repeat_interleave(torch.arange(len(batch_num_nodes),device=device), batch_num_nodes).to(device) #Sequentially assign nodes to batches

        batch_scores = model.forward(input=batch_x,A=A,x=x,batch=batch_indices,dropout=args.in_feat_dropout) #Model output
        
        batch_scores=batch_scores

        loss = get_loss(batch_scores, batch_targets)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.detach().item()

        epoch_train_mae += MAE(batch_scores, batch_targets)

        nb_data += batch_targets.size(0)
    
    epoch_loss /= (iter + 1)

    epoch_train_mae /= (iter + 1)

    return epoch_loss, epoch_train_mae, optimizer



#Evaluation/Test Method
def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device,dtype=torch.float32)
            batch_e = batch_graphs.edata['feat'].to(device,dtype=torch.float32)
            batch_targets = batch_targets.squeeze(-1).to(device,dtype=torch.float32)

            src,dest=batch_graphs.edges() #Get the src and destination indices

            indices=torch.stack([src, dest]) #edge indices for Adjacency matrix
    
            num_nodes=batch_graphs.num_nodes() #No of nodes in the batch
    
            A=Get_sparse_adjacency_matrix(indices,num_nodes,device)
            
            x=torch.rand(num_nodes,args.hidden,device=device,dtype=torch.float32) #Initialize the coordinates
            
            batch_num_nodes = batch_graphs.batch_num_nodes()  # Returns a tensor of node counts per graph
            
            batch_indices = torch.repeat_interleave(torch.arange(len(batch_num_nodes),device=device), batch_num_nodes).to(device) #Sequentially assign nodes to batches
    
            batch_scores = model.forward(input=batch_x,A=A,x=x,batch=batch_indices,dropout=args.in_feat_dropout) #Model output
            
            batch_scores=batch_scores

            loss = get_loss(batch_scores, batch_targets)

            epoch_test_loss += loss.detach().item()
           

            epoch_test_mae += MAE(batch_scores, batch_targets)

            nb_data += batch_targets.size(0)
        
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
    
    return epoch_test_loss, epoch_test_mae



def train_val_pipeline(train_loader,val_loader,test_loader,num_atom_type,num_bond_type):
    t0 = time.time()
    per_epoch_time = []
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if 'cuda' == args.device:
        torch.cuda.manual_seed(args.seed)
    
    print("Training Graphs: ", len(train_loader))
    print("Validation Graphs: ", len(val_loader))
    print("Test Graphs: ", len(test_loader))
    
    #Initialize the Custom GNN model
    model=Custom_GNN(L=args.layers,input_dim=args.hidden,hidden_dim=args.hidden,out_dim=args.hidden,dropout=args.dropout,device=args.device
                 ,flag=args.flag,readout=args.readout,dataset_name=args.dataset_name,act_fn=nn.ReLU(),emb_dim1=num_atom_type,emb_dim2=args.hidden)
    
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=args.lr_reduce_factor,
                                                     patience=args.lr_schedule_patience,
                                                     verbose=True)
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], []

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(args.epochs)) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, args.device, train_loader, epoch)

                epoch_val_loss, epoch_val_mae = evaluate_network(model, args.device, val_loader, epoch)
                _, epoch_test_mae = evaluate_network(model, args.device, test_loader, epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae,
                              test_MAE=epoch_test_mae)
                
                per_epoch_time.append(time.time()-start)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < args.min_lr:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time()-t0 > args.max_time*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(args.max_time))
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_mae = evaluate_network(model, args.device, test_loader, epoch)
    _, train_mae = evaluate_network(model, args.device, train_loader, epoch)
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    
    with open('Exec_Results' + '.txt', 'a') as f:
        f.write("""Dataset: {},\nModel: {}\n\n{}\n\nFINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\nConvergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(args.dataset_name,"Custom_GNN", model,
                  test_mae, train_mae, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
    
    return epoch_train_losses,epoch_val_losses,epoch_train_MAEs,epoch_val_MAEs


#Main method
if __name__=='__main__':
    train_loader,val_loader,test_loader,num_atom_type,num_bond_type=load_ZINC_Dataset(args.dataset_name,args.batch_size)
    
    epoch_train_losses,epoch_val_losses,epoch_train_MAEs,epoch_val_MAEs = train_val_pipeline(train_loader,val_loader,test_loader,num_atom_type,num_bond_type) #trigger the training and validation
    
    #Plot the performance
    fig,axis=plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    
    df=pd.DataFrame({
    'Epoch':np.array([e for e in range(1,len(epoch_train_losses)+1)]),
    'Train Performance':np.array(epoch_train_losses),
    'Validation Performance':np.array(epoch_val_losses),
    'Train Error':np.array(epoch_train_MAEs),
    'Validation Error':np.array(epoch_val_MAEs)
    })
    
    sns.lineplot(data=df,x='Epoch',y='Train Performance',ax=axis[0][0],label="Custom GNN")
    sns.lineplot(data=df,x='Epoch',y='Validation Performance',ax=axis[0][1],label='Custom GNN')
    sns.lineplot(data=df,x='Epoch',y='Train Error',ax=axis[1][0],label='Custom GNN')
    sns.lineplot(data=df,x='Epoch',y='Validation Error',ax=axis[1][1],label='Custom GNN')

    # Customizing the plot
    axis[0][0].set_title(args.dataset_name)
    axis[0][0].set_xlabel("Epoch")
    axis[0][0].set_ylabel("Training Loss")
    axis[0][0].tick_params(axis='x', rotation=45)
    axis[0][0].legend()
    axis[0][0].grid(True, linestyle='--', alpha=0.6)
    
    axis[0][1].set_title(args.dataset_name)
    axis[0][1].set_xlabel("Epoch")
    axis[0][1].set_ylabel("Validation Loss")
    axis[0][1].tick_params(axis='x', rotation=45)
    axis[0][1].legend()
    axis[0][1].grid(True, linestyle='--', alpha=0.6)

    axis[1][0].set_title(args.dataset_name)
    axis[1][0].set_xlabel("Epoch")
    axis[1][0].set_ylabel("Train MAE")
    axis[1][0].tick_params(axis='x', rotation=45)
    axis[1][0].legend()
    axis[1][0].grid(True, linestyle='--', alpha=0.6)

    axis[1][1].set_title(args.dataset_name)
    axis[1][1].set_xlabel("Epoch")
    axis[1][1].set_ylabel("Validation MAE")
    axis[1][1].tick_params(axis='x', rotation=45)
    axis[1][1].legend()
    axis[1][1].grid(True, linestyle='--', alpha=0.6)

    # Show plot
    plt.tight_layout()
    plt.savefig(f"/home/mlrl/Suranjan/Custom_GNN/plots/Zinc/Zinc Performance.png")
    plt.close()
