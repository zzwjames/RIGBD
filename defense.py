import imp
import time
import argparse
import numpy as np
import torch
torch.set_printoptions(threshold=10000)
from torch_geometric.datasets import Planetoid,Reddit2,Flickr
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated, clu_prune_unrelated_edge
import scipy.sparse as sp
from torch_geometric.utils import subgraph
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv','Citeseer','Reddit2'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--k', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--range', type=float, default=1.0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=40,
                    help="number of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none','reconstruct'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--weight_target', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--weight_ood', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--weight_targetclass', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--outter_size', type=int, default=4096,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=100,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.8,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')

parser.add_argument('--trigger_generator_address', type=str, default='./model_weights_cora.pth')
parser.add_argument('--poison_x', type=str, default='poison_x_cora.pt')
parser.add_argument('--poison_edge_index', type=str, default='poison_edge_index_cora.pt')
parser.add_argument('--poison_edge_weights', type=str, default='poison_edge_weights_cora.pt')
parser.add_argument('--poison_labels', type=str, default='poison_labels_cora.pt')
parser.add_argument('--idx_attach', type=str, default='Cora_UGBA.txt')

# GPU setting
parser.add_argument('--device_id', type=int, default=3,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
# args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)





#### dataset ####

from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
    


#### preprocess data ####

from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]



# from sklearn_extra import cluster
from models.backdoor import Backdoor
from models.construct import model_construct
from torch.distributions.bernoulli import Bernoulli
# from sklearn_extra import cluster
from models.backdoor import Backdoor
from models.construct import model_construct
from torch.distributions.bernoulli import Bernoulli

unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()

import ast
with open(args.idx_attach, 'r') as file:
    content = file.read()
    idx_attach = ast.literal_eval(content)

idx_attach = torch.tensor(idx_attach)
idx_attach = idx_attach.to(device)

# print(idx_attach)

#### load trigger generator and poisoned data ####
trigger_generator_address = args.trigger_generator_address
poison_x = torch.load(args.poison_x).to(device)
poison_edge_index = torch.load(args.poison_edge_index).to(device)
poison_edge_weights = torch.load(args.poison_edge_weights).to(device)
poison_labels = torch.load(args.poison_labels).to(device)



#### trigger generator ####
model = Backdoor(args,device)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx, trigger_generator_address, True)


#### #### 
mask = data.y[idx_attach] != args.target_class
mask = mask.to(device)
print('number of poisoned target nodes', mask.sum())
## only attack those has groud truth labels != target_class ##
idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
known_nodes = torch.cat([idx_train,idx_attach]).to(device)
predictions = []
# edge weight for clean edge_index, may use later #
edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)



#### train a backdoored model on poisoned graph #### 
test_model = model_construct(args,args.test_model,data,device).to(device) 
test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs, verbose=False)
test_model.eval()
clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)
output_clean = test_model(poison_x,poison_edge_index,poison_edge_weights)
ori_predict = torch.exp(output_clean[known_nodes])
# print("accuracy on poisoned target nodes: {:.4f}".format(clean_acc))

#### test backdoored model for comparison ####
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()

output = test_model(induct_x,induct_edge_index,induct_edge_weights)
flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
print("****Before Defense****")
print("ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
print("CA: {:.4f}".format(ca))


###### formal test ########

test_model = model_construct(args,args.test_model,data,device,add_self_loops=False).to(device) 
test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
test_model.eval()
clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)
output_clean = test_model(poison_x,poison_edge_index,poison_edge_weights)
ori_predict = torch.exp(output_clean[known_nodes])
print("accuracy on poisoned target nodes: {:.4f}".format(clean_acc))

drop_ratio = 0.5
def sample_noise_all(edge_index, edge_weight, device):
    # Ensure inputs are on the correct device
    edge_index = edge_index.to(device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device)
    else:
        edge_weight = edge_weight.to(device)

    # Generate mask for edge dropping
    drop_mask = Bernoulli(1 - drop_ratio).sample(edge_weight.size()).bool()

    # Apply mask to edges
    noisy_edge_index = edge_index[:, drop_mask]
    noisy_edge_weight = edge_weight[drop_mask]

    # Get node degrees
    node_degrees = torch.zeros(edge_index.max() + 1, device=device)
    node_degrees.index_add_(0, noisy_edge_index[0], torch.ones(noisy_edge_index.size(1), device=device))
    # print('degree', node_degrees)

    # Restore edges for isolated nodes
    isolated_nodes = node_degrees == 0
    # print('isolated_nodes', isolated_nodes)
    if isolated_nodes.any():
        potential_restore_edges = isolated_nodes[edge_index[0]]
        # print('potential_restore_edges', potential_restore_edges)
        restore_edges = edge_index[:, potential_restore_edges]
        noisy_edge_index = torch.cat([noisy_edge_index, restore_edges], dim=1)
        restored_weights = torch.ones(restore_edges.size(1), device=device)
        noisy_edge_weight = torch.cat([noisy_edge_weight, restored_weights], dim=0)

    return noisy_edge_index, noisy_edge_weight
# def sample_noise_all(edge_index, edge_weight,device):
#     noisy_edge_index = edge_index.clone().detach()
#     if(edge_weight == None):
#         noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
#     else:
#         noisy_edge_weight = edge_weight.clone().detach()
#     # # rand_noise_data = copy.deepcopy(data)
#     # rand_noise_data.edge_weight = torch.ones([rand_noise_data.edge_index.shape[1],]).to(device)
#     m = Bernoulli(torch.tensor([1-drop_ratio]).to(device))
#     mask = m.sample(noisy_edge_weight.shape).squeeze(-1).int()
#     # print('mask',mask)
#     rand_inputs = torch.randint_like(noisy_edge_weight, low=0, high=2).squeeze().int().to(device)
#     # print(rand_noise_data.edge_weight.shape,mask.shape)
#     noisy_edge_weight = noisy_edge_weight * mask #+ rand_inputs * (1-mask)
        
#     if(noisy_edge_weight!=None):
#         noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
#         noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
#     return noisy_edge_index, noisy_edge_weight

predictions = []
K=20
for i in range(K):
            test_model.eval()
            noisy_poison_edge_index, noisy_poison_edge_weights = sample_noise_all(poison_edge_index, poison_edge_weights, device)
            output = test_model(poison_x,noisy_poison_edge_index,noisy_poison_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
            train_clean_rate = (output.argmax(dim=1)[idx_train]==data.y[idx_train]).float().mean()
            predictions.append(torch.exp(output[known_nodes]))

epsilon = 1e-8
deviations = []
for sub_pred in predictions:
    sub_pred += epsilon
    deviation = F.kl_div(sub_pred.log(), ori_predict, reduce=False)
    deviations.append(deviation)

summed_deviations = torch.zeros_like(deviations[0]).to(deviations[0].device)
for deviation in deviations:
    ##### summed deviations for each node #####
    summed_deviations += deviation


##### get the index for nodes with less robustness #####
    
##### args.vs_number is unknown #####
# index_of_less_robust = torch.sort(torch.mean(summed_deviations,dim=-1),descending=True)[1][:mask.sum()]
index_of_less_robust = torch.sort(torch.mean(summed_deviations,dim=-1),descending=True)[1]

def find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, target_class):
    # Get the specific list to iterate through
    labels_list = poison_labels[bkd_tn_nodes[index_of_less_robust]]

    # Iterate through the list with index
    for i in range(len(labels_list) - 1):  # -1 to avoid index out of range
        if labels_list[i] != target_class and labels_list[i + 1] != target_class:
            return i - 1

    # Return None if the condition is not met in the loop
    return None

# Example usage:
# Assuming poison_labels, bkd_tn_nodes, index_of_less_robust, and target_class are defined
result_index = find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, args.target_class)
print("Index found:", result_index)
# print(poison_labels[bkd_tn_nodes[index_of_less_robust][:result_index]])

indexs = poison_labels[bkd_tn_nodes[index_of_less_robust][:result_index-1]]
count = 0
for i in indexs:
    if i == args.target_class:
        count += 1
## correct
correct = count

## fasle
false = len(indexs) - count



test_model = model_construct(args,args.test_model,data,device).to(device) 

# test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False, finetune=True, attach=bkd_tn_nodes[index_of_less_robust][:result_index])
test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=400,verbose=False, finetune=True, attach=bkd_tn_nodes[index_of_less_robust][:result_index])
## test model on purified graph (poisoned target node) ##
# clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)

induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
# induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_attach,poison_x,induct_edge_index,induct_edge_weights,device)
induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()

output = test_model(induct_x,induct_edge_index,induct_edge_weights)
flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
print("****After Defense****")
print('recall', correct/mask.sum())
print('Accuracy', correct/len(indexs))
print("ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
print("CA: {:.4f}".format(ca))



