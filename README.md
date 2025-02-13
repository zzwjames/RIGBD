## Robustness Inspired Graph Backdoor Defense [ICLR 25 Oral] [[paper]](https://openreview.net/forum?id=trKNi4IUiP&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))


**1.** Download the weights, poisoned graph, and idx_attach from here: https://www.dropbox.com/scl/fo/iw2nouqgqx7nz7y32043w/AMAJNKdEMfBz0v-H6omERS0?rlkey=2d3ane6ak6kzhv79xsvhx2j72&e=1&st=4d7rd61z&dl=0

  
  

**2.** Modify the address for these **6** args: '**(1) trigger_generator_address**', '**(2) poison_x**', '**(3) poison_edge_index**', '**(4) poison_edge_weights**', '**(5) poison_labels'**, '**(6) idx_attach**'

## RUN EXAMPLES

 

**Cora**

  

    python defense.py --dataset Cora --vs_number 40 --trigger_generator_address './model_weights_cora.pth' --poison_x 'poison_x_cora.pt' --poison_edge_index 'poison_edge_index_cora.pt' --poison_edge_weights 'poison_edge_weights_cora.pt' --poison_labels 'poison_labels_cora.pt' --idx_attach 'Cora_UGBA.txt'

  

**PubMed**

  

    python defense.py --dataset Pubmed --vs_number 160 --trigger_generator_address './model_weights_pubmed.pth' --poison_x 'poison_x_pubmed.pt' --poison_edge_index 'poison_edge_index_pubmed.pt' --poison_edge_weights 'poison_edge_weights_pubmed.pt' --poison_labels 'poison_labels_pubmed.pt' --idx_attach 'PubMed_DPGBA.txt'

  

**OGB-arxiv**

  

    python defense.py --dataset ogbn-arxiv --vs_number 565 --trigger_generator_address './model_weights_arxiv.pth' --poison_x 'poison_x_arxiv.pt' --poison_edge_index 'poison_edge_index_arxiv.pt' --poison_edge_weights 'poison_edge_weights_arxiv.pt' --poison_labels 'poison_labels_arxiv.pt' --idx_attach 'OGBArxiv_UGBA.txt'


If you find this repo to be useful, please consider cite our paper. Thank you.

    @inproceedings{
    zhang2025robustness,
    title={Robustness Inspired Graph Backdoor Defense},
    author={Zhiwei Zhang and Minhua Lin and Junjie Xu and Zongyu Wu and Enyan Dai and Suhang Wang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=trKNi4IUiP}
    }
