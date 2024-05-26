Cora: 40
Pubmed: 160
Arxiv: 565

**1.** Download the weights and idx_attach from here: https://www.dropbox.com/scl/fo/iw2nouqgqx7nz7y32043w/AMAJNKdEMfBz0v-H6omERS0?rlkey=2d3ane6ak6kzhv79xsvhx2j72&e=1&st=4d7rd61z&dl=0


**2.**
Modify the address for these 5 args: 'trigger_generator_address', 'poison_x', 'poison_edge_index', 'poison_edge_weights', 'poison_labels', 'idx_attach'

**Cora**

python defense.py --dataset Cora --vs_number 40 --trigger_generator_address './model_weights_cora.pth' --poison_x 'poison_x_cora.pt' --poison_edge_index 'poison_edge_index_cora.pt' --poison_edge_weights 'poison_edge_weights_cora.pt' --poison_labels 'poison_labels_cora.pt' --idx_attach 'Cora_UGBA.txt'

**PubMed**

python defense.py --dataset Pubmed --vs_number 160

**OGB-arxiv**

python defense.py --dataset ogbn-arxiv --vs_number 565


