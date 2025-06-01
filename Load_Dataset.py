from torch_geometric.datasets import Planetoid,Amazon,QM9
from torch_geometric.transforms import NormalizeFeatures
from scipy.sparse import coo_matrix
from collections import defaultdict
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as DL
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl,PygGraphPropPredDataset
from torch.utils.data import DataLoader,Subset
from data.data import LoadData
from torch_geometric.data import Data
import numpy as np
import torch
import pickle
from Utils import Get_sparse_adjacency_matrix


#Method that is used to convert the sparse matrix to sparse torch tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    # Ensure the input is a COO matrix
    if not isinstance(sparse_mx, coo_matrix):
        sparse_mx = coo_matrix(sparse_mx)
    
    # Convert to numpy float32 and extract row, col, and data
    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    # Construct a PyTorch sparse tensor
    return torch.sparse.FloatTensor(indices, values, shape)



#Method that is used to generate the train validation and test_split for Amazon and Co-author dataset
def Generate_test_train_validation_mask(data,num_train_per_class,num_val_per_class,device):
    num_nodes = data.x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    validation_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    
    # Group nodes by label
    label_indices = defaultdict(list)
    labels = data.y.cpu().numpy()  # Convert to numpy for easier indexing

    for i, label in enumerate(labels):
        label_indices[int(label)].append(i)

    # Sample train indices
    train_set = set()
    for indices in label_indices.values():
        sample_size = min(num_train_per_class, len(indices))  # Handle small classes
        train_set.update(np.random.choice(indices, size=sample_size, replace=False))

    # Exclude train samples from validation/test selection
    remaining_samples = [i for i in range(num_nodes) if i not in train_set]

    # Group remaining nodes by label for validation selection
    label_indices2 = defaultdict(list)
    for index in remaining_samples:
        label_indices2[int(labels[index])].append(index)

    # Print class distribution
    '''for label, indices in label_indices2.items():
        print(f"Class {label}: {len(indices)} remaining samples")'''

    # Sample validation indices
    validation_set = set()
    for indices in label_indices2.values():
        sample_size = min(num_val_per_class, len(indices))  # Handle small classes
        validation_set.update(np.random.choice(indices, size=sample_size, replace=False))

    # Remaining samples are test samples
    test_set = set(remaining_samples) - validation_set

    # Sanity checks
    assert len(train_set & validation_set) == 0, "Train and validation sets overlap!"
    assert len(train_set & test_set) == 0, "Train and test sets overlap!"
    assert len(validation_set & test_set) == 0, "Validation and test sets overlap!"

    print(f"Train: {len(train_set)}, Validation: {len(validation_set)}, Test: {len(test_set)}")

    # Set masks
    train_mask[list(train_set)] = True
    validation_mask[list(validation_set)] = True
    test_mask[list(test_set)] = True

    return train_mask,validation_mask,test_mask


#Method that is used to load the Planetoid Datasets
def load_Planetoid_Dataset(device,name):
    dataset=Planetoid(root='/home/mlrl/Suranjan/Graph_Datasets',name=name,transform=NormalizeFeatures())
    print('Loading {} dataset...'.format(dataset))
    data=dataset[0]

    features = data.x.to(device)  # No need for sparse -> dense conversions
    labels=data.y.to(device=device) #Labels (Ground Truth)

    #build the graph
    #Get the unordered edges from pytorch geometric and reshape
    edge_unordered=data.edge_index.numpy().T
    
    #Mapping the node indices to have a contigous order
    idx=np.arange(data.num_nodes)
    idx_mapping={j:i for i,j in enumerate(idx)}
    edges=np.array(list(map(idx_mapping.get,edge_unordered.flatten())),dtype=np.int32).reshape(edge_unordered.shape)
    
    #Create the adjacency matrix
    adj_matrix=coo_matrix(
        (np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),
         shape=(data.num_nodes,data.num_nodes),
         dtype=np.float32)
    
    #Used to add the reverse edges in case the graph is directed
    adj = adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix) - adj_matrix.multiply(adj_matrix.T > adj_matrix)
    adj=sparse_mx_to_torch_sparse_tensor(adj).to(device)

    idx_train = data.train_mask
    idx_val = data.val_mask
    idx_test = data.test_mask

    return adj, features, dataset, labels, idx_train, idx_val, idx_test

#Method that is used to load the Amazon Datasets
def load_Amazon_Dataset(device,name,train_size,val_size):
    dataset=Amazon(root='/home/mlrl/Suranjan/Graph_Datasets',name=name,transform=NormalizeFeatures())
    print('Loading {} dataset...'.format(dataset))
    data=dataset[0]

    features = data.x.to(device)  # No need for sparse -> dense conversions
    
    labels=data.y.to(device=device)

    #build the graph
    #Get the unordered edges from pytorch geometric and reshape
    edge_unordered=data.edge_index.numpy().T
    
    #Mapping the node indices to have a contigous order
    idx=np.arange(data.num_nodes)
    idx_mapping={j:i for i,j in enumerate(idx)}
    edges=np.array(list(map(idx_mapping.get,edge_unordered.flatten())),dtype=np.int32).reshape(edge_unordered.shape)
    
    #Create the adjacency matrix
    adj_matrix=coo_matrix(
        (np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),
         shape=(data.num_nodes,data.num_nodes),
         dtype=np.float32)
    
    #Used to add the reverse edges in case the graph is directed
    adj = adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix) - adj_matrix.multiply(adj_matrix.T > adj_matrix)
    adj=sparse_mx_to_torch_sparse_tensor(adj).to(device)

    #data,num_train_per_class=20,num_test_instance=1000
    idx_train,idx_val,idx_test=Generate_test_train_validation_mask(data,train_size,val_size,device)
    
    
    #idx_train = data.train_mask
    #idx_val = data.val_mask
    #idx_test = data.test_mask

    return adj, features, dataset, labels, idx_train, idx_val, idx_test


#Method that is used to load the QM9 dataset
def load_QM9_Dataset(device,train_set_size,test_pct,batch_size):
    dataset=QM9(root='/home/mlrl/Suranjan/Graph_Datasets')

    print('Loading {} dataset...'.format(dataset))

    # Train/Test/Validation split sizes (as per EGNN paper)
    train_size = train_set_size # ~100K (default)
    test_size = int(test_pct * len(dataset))  # ~13K (default)
    validation_size = len(dataset) - (train_size + test_size)  # ~18K (default)

    print(f"Training Size: {train_size}, Test Size: {test_size}, Validation Size: {validation_size}")

    #Randomly split the dataset
    train_set,test_set,validation_Set=random_split(dataset,[train_size,validation_size,test_size])
     
    #Prepare the dataloaders 
    train_loader=DL(train_set,batch_size=batch_size,shuffle=True)
    test_loader=DL(test_set,batch_size=batch_size,shuffle=False)
    validation_loader=DL(validation_Set,batch_size=batch_size,shuffle=False)

    return train_loader,validation_loader,test_loader



def dgl_to_pyg(dgl_graph, y):
    """Convert a DGL graph to a PyG Data object."""
    edge_index = torch.stack(dgl_graph.edges())  # Convert DGL edges to PyG format
    x = dgl_graph.ndata["feat"] if "feat" in dgl_graph.ndata else None  # Node features
    return Data(x=x, edge_index=edge_index, y=y)


#Method that is used to get the ogbg dataloaders for a particular MoleculeNet dataset
def load_OGBG_Dataset(dataset_name:str,batch_size:int) -> tuple:
    dataset = DglGraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split() 
    
    # Convert each DGL graph to PyG Data format
    pyg_dataset = [dgl_to_pyg(dataset[i][0], dataset[i][1]) for i in range(len(dataset))]

    train_loader = DL(Subset(pyg_dataset,split_idx["train"]), batch_size=batch_size, shuffle=True)
    valid_loader = DL(Subset(pyg_dataset,split_idx["valid"]), batch_size=batch_size, shuffle=False)
    test_loader =  DL(Subset(pyg_dataset,split_idx["test"]), batch_size=batch_size, shuffle=False)
    num_classes=int(dataset.num_classes)
    eval_metric=dataset.eval_metric
    return train_loader,valid_loader,test_loader,num_classes,eval_metric


def load_OGBG_Dataset2(dataset_name:str,batch_size:int) -> tuple:
    dataset = DglGraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split() 
    

    train_loader = DataLoader(Subset(dataset,split_idx["train"]), batch_size=batch_size, shuffle=True,collate_fn=collate_dgl)
    valid_loader = DataLoader(Subset(dataset,split_idx["valid"]), batch_size=batch_size, shuffle=False,collate_fn=collate_dgl)
    test_loader =  DataLoader(Subset(dataset,split_idx["test"]), batch_size=batch_size, shuffle=False,collate_fn=collate_dgl)
    num_classes=int(dataset.num_classes)
    eval_metric=dataset.eval_metric
    return train_loader,valid_loader,test_loader,num_classes,eval_metric



#Method that is used to load the Zinc dataset
def load_ZINC_Dataset(DATASET_NAME,batch_size):
    dataset = LoadData(DATASET_NAME)
    trainset,valset,testset=dataset.train,dataset.val,dataset.test
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
    num_atom_type= dataset.num_atom_type # known meta-info about the zinc dataset 28
    num_bond_type= dataset.num_bond_type # known meta-info about the zinc dataset; 4
    return train_loader,val_loader,test_loader,num_atom_type,num_bond_type