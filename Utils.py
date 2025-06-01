# helper function to visualize node embeddings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from typing import Iterator
import torch


#Method that is used to visualize the TSNE plots
def visualize(h, color,path):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(f"{path}.png")
    plt.close()



#Method used to train the model
def train(model,input,train_mask,labels,optimizer,A):
    model.train()  #Set the model in training mode
    optimizer.zero_grad() #Reset the gradients
    logits=model(input,A) #Get the logits for all nodes
    train_loss=F.nll_loss(logits[train_mask],labels[train_mask])
    train_loss.backward() #Backprop
    #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #Gradient clipping
    optimizer.step() #update
    return train_loss

#Method that will be used to test the model
def test(model,input,A,test_mask,labels):
    model.eval() #model set to evaluation mode
    logits=model(input,A) #Get the logits for all nodes
    loss=F.nll_loss(logits[test_mask],labels[test_mask]) #Get the loss
    pred = logits.argmax(dim=1) #Derive the Predictions from labels
    y_true=labels[test_mask].cpu().numpy()
    y_pred=pred[test_mask].cpu().numpy()
    acc_test = accuracy_score(y_true,y_pred)
    return acc_test,loss


#Method used to calculate the Mean and the MAD of the labels for QM9 dataset
def compute_mean_mad(loader:Iterator,index:int) -> tuple:
    all_labels=[] #Store all the labels
    for batch in loader:
        label=batch.y[:,index]
        all_labels.append(label)
    values=torch.cat(all_labels,dim=0) #Concatenate along the row
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

#Method that is used to get the sparse adjacency matrix from the dataset
def Get_sparse_adjacency_matrix(indices:torch.tensor,num_nodes:int,device:str) -> torch.sparse.Tensor:
    indices=indices.to(device) #Load the indices to device
    values=torch.ones([indices.shape[1]],device=device) #Set the values to all ones
    shape=(num_nodes,num_nodes) #n,n
    return torch.sparse_coo_tensor(indices,values,shape,dtype=torch.float32)