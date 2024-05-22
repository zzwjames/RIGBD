
#!/usr/bin/env python
# coding: utf-8

# In[1]: 


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
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Pubmed', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv','Citeseer','Reddit2'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=2)
parser.add_argument('--k', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--rec_epochs', type=int,  default=30, help='Number of epochs to train benign and backdoor model.')
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
parser.add_argument('--defense_mode', type=str, default="prune",
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
parser.add_argument('--evaluate_mode', type=str, default='overall',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=2,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'Reddit2'):
    dataset = Reddit2(root='./data/Reddit2/', \
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
    
if(args.dataset == 'Reddit2'):
    num_nodes_to_sample = 20000  # Adjust this based on your needs

    # Randomly select a subset of nodes
    sampled_nodes = torch.randint(data.num_nodes, (num_nodes_to_sample,), device=device)

    # Perform subgraph sampling
    edge,_ = subgraph(sampled_nodes, data.edge_index)
    data.edge_index = edge
# we build our own train test split 
#%% 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


# In[9]:

from sklearn_extra import cluster
from models.backdoor import Backdoor
from models.construct import model_construct
import heuristic_selection as hs
from torch.distributions.bernoulli import Bernoulli

def sample_noise_all(edge_index, edge_weight,device):
    noisy_edge_index = edge_index.clone().detach()
    if(edge_weight == None):
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    else:
        noisy_edge_weight = edge_weight.clone().detach()
    # # rand_noise_data = copy.deepcopy(data)
    # rand_noise_data.edge_weight = torch.ones([rand_noise_data.edge_index.shape[1],]).to(device)
    m = Bernoulli(torch.tensor([0.5]).to(device))
    mask = m.sample(noisy_edge_weight.shape).squeeze(-1).int()
    # print('mask',mask)
    rand_inputs = torch.randint_like(noisy_edge_weight, low=0, high=2).squeeze().int().to(device)
    # print(rand_noise_data.edge_weight.shape,mask.shape)
    noisy_edge_weight = noisy_edge_weight * mask #+ rand_inputs * (1-mask)
        
    if(noisy_edge_weight!=None):
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    return noisy_edge_index, noisy_edge_weight


# from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
if(args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
# print("#Attach Nodes:{}".format(size))
assert size>0, 'The number of selected trigger nodes must be larger than 0!'
# here is randomly select poison nodes from unlabeled nodes
if(args.selection_method == 'none'):
    idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'cluster'):
    idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    if(args.dataset == 'Pubmed'):
        idx_attach = hs.cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    else:
        idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
print("idx_attach: {}".format(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
print('unlabeled_idx',len(unlabeled_idx))
# In[10]:
# train trigger generator 
model = Backdoor(args,device)


###### get the pretrained trigger generator #####
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx, True)
###### #####


# model.trojan.load_state_dict(torch.load('model_weights.pth'))
# model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
# poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()
# print('saving')
# torch.save(poison_x,'poison_x.pt')
# torch.save(poison_edge_index,'poison_edge_index.pt')
# torch.save(poison_edge_weights,'poison_edge_weights.pt')
# torch.save(poison_labels,'poison_labels.pt')
# print('idx_attach',idx_attach)

##### get poisoned dataset #####
poison_x = torch.load('poison_x.pt')
poison_edge_index = torch.load('poison_edge_index.pt')
poison_edge_weights = torch.load('poison_edge_weights.pt')
poison_labels = torch.load('poison_labels.pt')
#####-------------- #####

bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)


models = ['GCN']
total_overall_asr = 0
total_overall_ca = 0
##### we train a test model on the poisoned dataset, to let it be backdoored #####
for test_model in models:
    args.test_model = test_model
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=1)
    overall_asr = 0
    overall_ca = 0
    for seed in seeds:
        args.seed = seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print(args)
        #%%
        test_model = model_construct(args,args.test_model,data,device).to(device) 
        known_nodes = torch.cat([idx_train,idx_attach]).to(device)
        predictions = []
        # test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
        # print(data.y[idx_attach])
        print('poison_labels[idx_attach]',poison_labels[idx_attach])
        edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        test_model.fit(data.x, data.edge_index, edge_weight, data.y, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
        test_model.eval()
        output, x = test_model(poison_x,poison_edge_index,poison_edge_weights)
        ori_predict = torch.exp(output[known_nodes])

        # for id in idx_attach:
        #     print(poison_labels[id])
        #     print(data.y[id])


        ##### random drop edges for 20 times, and record the predictions by fixed backdoored model #####
        for i in range(20):
            test_model.eval()
            noisy_poison_edge_index, noisy_poison_edge_weights = sample_noise_all(poison_edge_index, poison_edge_weights, device)
            output, x = test_model(poison_x,noisy_poison_edge_index,noisy_poison_edge_weights)
            # torch.save(test_model, 'model.pth')
            train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
            train_clean_rate = (output.argmax(dim=1)[idx_train]==data.y[idx_train]).float().mean()
            # predictions.append(output.argmax(dim=1)[known_nodes])
            predictions.append(torch.exp(output[known_nodes]))
            # print(output.argmax(dim=1)[idx_attach])
            # print("target class rate on Vs: {:.4f}".format(train_attach_rate))
            # print("clean rate on Vs: {:.4f}".format(train_clean_rate))
        #####------------#####
            
        ##### predictions: 20xnum_nodesxnum_class
        ##### compared predictions difference to the original prediction ##### 
        deviations = []
        for sub_pred in predictions:
            # Calculate deviation as 1 where absolute difference is greater than 0, else 0
            # print('sub_pred',sub_pred)
            # print('ori_predict',ori_predict)
            deviation = F.kl_div(sub_pred.log(), ori_predict, reduce=False)
            deviations.append(deviation)
        # print('deviations',deviations)
        summed_deviations = torch.zeros_like(deviations[0]).to(deviations[0].device)
        for deviation in deviations:
            ##### summed deviations for each node #####
            summed_deviations += deviation

        # print('summed_deviations',torch.mean(summed_deviations,dim=-1))
        # print('target_deviations',torch.sort(torch.mean(summed_deviations,dim=-1)[-40:],descending=True)[0])
        # print('max_clean_deviation',torch.sort(torch.mean(summed_deviations,dim=-1)[:-40],descending=True)[0][:10])
        
        ##### get the index for nodes with less robustness #####
        index_of_less_robust = torch.sort(torch.mean(summed_deviations,dim=-1),descending=True)[1][:40]
        print('index_of_less_robust',index_of_less_robust)

        ##### count how many poisoned target nodes are selected in less robustness nodes #####
        count = 0
        dd = []
        for idx in index_of_less_robust:
            if idx >= len(known_nodes)-40:
                count += 1
                # print(data.y[idx])
        print('count',count)
        ##### #####

        #####  #####
        a=torch.mean(summed_deviations,dim=-1)
        a_excluding_last_40 = a[:-40].detach().cpu()
        a_last=a[-40:].detach().cpu()

        a_last_40 = a[-40:].detach().cpu()

        indices_to_remove = [i for i, node in enumerate(known_nodes[-40:]) if data.y[node] == 2]

        indices_to_remove = []

        a_last_list = list(a_last)

        indices_to_check = range(len(known_nodes[-40:]))

        indices_to_remove = []
        
        ##### find those nodes that are already have target class or poisoned failed #####
        for i in indices_to_check:
            condition1 = data.y[known_nodes[-40:][i]] == 2
            condition2 = output.argmax(dim=1)[idx_attach[i]] != args.target_class
            
            if condition1 or condition2:
                indices_to_remove.append(i)

        for index in sorted(indices_to_remove, reverse=True):
            del a_last_list[index]
        

        a_last_40 = a_last_list
        plt.figure(figsize=(10, 6))
        ##### benign nodes #####
        plt.hist(a_excluding_last_40, bins=20, alpha=0.5, label='a[:-40]', density=True)
        ##### poisoned success nodes #####
        plt.hist(a_last_40, bins=20, alpha=0.5, label='a[-40:]',density=True)
        plt.legend(loc='upper right')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of a[:-40] and a[-40:]')
        plt.savefig('a.jpg')
        plt.show()

        #####  retrain the test model on clean graph  #####

        # test_model = model_construct(args,args.test_model,data,device).to(device) 
        # edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        # test_model.fit(data.x, data.edge_index, edge_weight, data.y, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
        
        # induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        # induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        
        # clean_acc = test_model.test(data.x,data.edge_index, edge_weight,data.y,idx_attach)
        # print('clean ', clean_acc)
        # known_nodes = torch.cat([idx_train,idx_attach]).to(device)
        # predictions = []
        # bkd_tn_nodes = idx_train
        # test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
        # test_model.eval()

        

       
        #%%
        induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


        if(args.evaluate_mode == '1by1'):
            from torch_geometric.utils  import k_hop_subgraph
            overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
            asr = 0
            flip_asr = 0
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            for i, idx in enumerate(flip_idx_atk):
                idx=int(idx)
                sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                ori_node_idx = sub_induct_nodeset[sub_mapping]
                relabeled_node_idx = sub_mapping
                sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                with torch.no_grad():
                    # inject trigger on attack test nodes (idx_atk)'''
                    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device)
                    
                    # pattern = torch.tensor(data.x[idx_train][:,-20:].mean(dim=0),device=device)
                    # induct_x[:, -20:] = pattern
                    # torch.cat((induct_x,pattern.repeat(len(induct_x),1)),dim=1)
                    
                    
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                    # # do pruning in test datas'''
                    # torch.save(induct_x,'tensor.pt')
                    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,False)
                    # attack evaluation
                    # else:
                    #     induct_edge_index,induct_edge_weights = clu_prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,False)
                    # print('prune over')
                    output, x = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                    asr += train_attach_rate
                    if(data.y[idx] != args.target_class):
                        flip_asr += train_attach_rate
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
                    output = output.cpu()
            asr = asr/(idx_atk.shape[0])
            flip_asr = flip_asr/(flip_idx_atk.shape[0])
            print("Overall ASR: {:.4f}".format(asr))
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
        elif(args.evaluate_mode == 'overall'):
            # %% inject trigger on attack test nodes (idx_atk)'''
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
            # do pruning in test datas'''
            if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
            # attack evaluation
            
            # cluster
            
                
            output, x = test_model(induct_x,induct_edge_index,induct_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
            print("ASR: {:.4f}".format(train_attach_rate))
            asr = train_attach_rate
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
            ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
            print("CA: {:.4f}".format(ca))

            induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
            output = output.cpu()

        overall_asr += asr
        overall_ca += clean_acc

        test_model = test_model.cpu()
        
    overall_asr = overall_asr/len(seeds)
    overall_ca = overall_ca/len(seeds)
    print("Overall ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, args.test_model, args.seed))
    print("Overall Clean Accuracy: {:.4f}".format(overall_ca))

    total_overall_asr += overall_asr
    total_overall_ca += overall_ca
    test_model.to(torch.device('cpu'))
    torch.cuda.empty_cache()
total_overall_asr = total_overall_asr/len(models)
total_overall_ca = total_overall_ca/len(models)
print("Total Overall ASR: {:.4f} ".format(total_overall_asr))
print("Total Clean Accuracy: {:.4f}".format(total_overall_ca))