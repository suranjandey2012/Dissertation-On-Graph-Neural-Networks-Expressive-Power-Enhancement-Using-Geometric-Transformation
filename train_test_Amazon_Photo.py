from Load_Dataset import load_Amazon_Dataset
from Params import Get_Train_Settings
from Utils import train,test,visualize
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from model import Custom_GNN
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

#Get the training related settings for the Cora Dataset
args=Get_Train_Settings()

#Set the seed for the model
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


#Load the dataset
adj, features, dataset, labels, idx_train, idx_val, idx_test = load_Amazon_Dataset(args.device,'Photo',args.train_set_size,args.val_set_size)
adj=adj.to(args.device)
features=features.to(args.device)
labels=labels.to(args.device)
idx_train=idx_train.to(args.device)
idx_val=idx_val.to(args.device)
idx_test=idx_test.to(args.device)

print("Dataset Details")
print("-".join([' ' for _ in range(30)]))
print(f"Node Features Dimension:{features.shape}")
print(f"Adjacency Matrix Dimension:{adj.shape}")
print(f"Distinct labels:{torch.unique(labels)}")
print(f"Number of training samples:{len(idx_train[idx_train])}")
print(f"Number of validation samples:{len(idx_val[idx_val])}")
print(f"Number of test samples:{len(idx_test[idx_test])}")

#Set the input and output dimension of the model
input_dim=features.shape[1]
out_dim=dataset.num_classes
num_nodes=dataset[0].num_nodes


#Define the model and the optimizer
model=Custom_GNN(L=args.layers,input_dim=input_dim,hidden_dim=args.hidden,out_dim=out_dim,dropout=args.dropout,device=args.device) #Model Definition
optimizer=Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay) #Define the Adam optimizer

#Visualize the data from the untrained model
out=model(features,adj)
visualize(out,labels.detach().cpu().numpy(),"/home/mlrl/Suranjan/Custom_GNN/plots/Amazon_Photo/Features Before Training")

#List used to store the training loss and the validation accuracy
Training_loss=[] #List used to store the training loss for individual epoches
Validation_Accuracy=[] #List used to store the validation accuracy of the model
Validation_loss=[] #List used to store the validation loss
print(f"Training Started")
print('-'.join('' for _ in range(100)))
for e in tqdm(range(args.epochs),desc="Training Progress"):
    best_val_acc=0
    loss=train(model,features,idx_train,labels,optimizer,adj) #Get the training loss for current epoch
    val_acc,val_loss=test(model,features,adj,idx_val,labels)  #Run the validation

    tqdm.write(f"Epoch:{e}||Training Loss:{loss:.4f}||Validation Accuarcy:{val_acc:.4f}") #Print the loss and the validation accuracy for each epoch
    
    if(val_acc>best_val_acc):          #Save the model that returns the best validation accuracy
        best_val_acc = val_acc
        best_model=model.state_dict()
    
    Training_loss.append(loss.item())   #Store the loss and accuracy values in a list for visualization
    Validation_loss.append(val_loss.item())
    Validation_Accuracy.append(val_acc)

#Evaluate the model
final_model=Custom_GNN(L=args.layers,input_dim=input_dim,hidden_dim=args.hidden,out_dim=out_dim,dropout=args.dropout,device=args.device)
final_model.load_state_dict(best_model)


final_model.eval()
test_acc,_=test(final_model,features,adj,idx_test,labels)

# Write the test accuracy to a file
with open("Exec_Results2.txt", "a") as f:
    f.write(f"Test Accuracy: {test_acc*100:.4f}\n")


#Visualize the embeddings after training the model
final_model.eval()
out=final_model(features,adj)

# visualizing node embeddings after training
visualize(out,labels.detach().cpu().numpy(),'/home/mlrl/Suranjan/Custom_GNN/plots/Amazon_Photo/Custom_GNN Final TSNE')


df=pd.DataFrame({
    'Epoch':np.array([e+1 for e in range(args.epochs)]),
    'Validation_Accuracy':np.array(Validation_Accuracy),
    'Training_Loss':np.array(Training_loss),
    'Validation_Loss':np.array(Validation_loss)
})

#Visualize the plots
fig,axis=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
sns.lineplot(data=df,x='Epoch',y='Training_Loss',ax=axis[0],label="Training Loss")
sns.lineplot(data=df,x='Epoch',y='Validation_Loss',ax=axis[0],label='Validation Loss')
sns.lineplot(data=df,x='Epoch',y='Validation_Accuracy',ax=axis[1],label='Validation Accuracy')

# Customizing the plot
axis[0].set_title("Loss vs. Epoch For Amazon_Photo Dataset")
axis[0].set_xlabel("Epoch")
axis[0].set_ylabel("Loss")
axis[0].tick_params(axis='x', rotation=45)
axis[0].legend()
axis[0].grid(True, linestyle='--', alpha=0.6)  # Add grid lines for better visualization


axis[1].set_title("Validation Accuracy vs. Epoch For Amazon_Photo Dataset")
axis[1].set_xlabel("Epoch")
axis[1].set_ylabel("Validation Accuracy")
axis[1].tick_params(axis='x', rotation=45)
axis[1].legend()
axis[1].grid(True, linestyle='--', alpha=0.6)  # Add grid lines

# Show plot
plt.tight_layout()
plt.savefig(f"/home/mlrl/Suranjan/Custom_GNN/plots/Amazon_Photo/Custom_GNN Training and Validation Accuracy Against Number Of Epochs.png")
plt.close()
