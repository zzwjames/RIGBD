# models=(GCN GraphSage GAT)
models=(GCN)
# isolate means the Prune+LD defense method
# defense_modes=(none prune isolate)
defense_modes=(none)
# SBA: Pubmed
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_SBA.py \
#             --prune_thrd=0.2\
#             --attack_method=Rand_Gene\
#             --vs_size=40\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=Pubmed
#     done    
# done

# GTA: Pubmed
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
        # python -u run_GTA.py \
        #     --prune_thr=0.2\
        #     --vs_size=160\
        #     --test_model=GCN\
        #     --defense_mode=none\
        #     --epochs=200\
        #     --dataset=Pubmed
#     done    
# done

# # GTA: Cora
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
        # python -u run_GTA.py \
        #     --prune_thr=0.1\
        #     --vs_size=40\
        #     --test_model=GCN\
        #     --defense_mode=none\
        #     --epochs=200\
        #     --dataset=Cora
#     done    
# done

# # Rand: Cora
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_bkd_baseline.py \
#             --prune_thr=0.1\
#             --attack_method=Rand_Gene\
#             --vs_size=10\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=Cora
#     done    
# done

# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_SBA.py \
#             --prune_thr=0.1\
#             --attack_method=Rand_Samp\
#             --vs_size=10\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=Cora
#     done    
# done

# # GTA: Flickr
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_GTA.py \
#             --train_lr 0.02 \
#             --hidden 256 \
#             --prune_thr=0.4\
#             --vs_size=120\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --target_class=6\
#             --dataset=Flickr
#     done    
# done

# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
        python -u run_GTA.py \
            --train_lr 0.02 \
            --prune_thr=0.4\
            --vs_size=565\
            --test_model=GCN\
            --defense_mode=none\
            --epochs=200\
            --target_class=2\
            --trojan_epochs=200\
            --dataset=ogbn-arxiv
#     done    
# done

# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_SBA.py \
#             --train_lr 0.02 \
#             --hidden 64 \
#             --prune_thr=0.4\
#             --attack_method=Rand_Gene\
#             --vs_size=160\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=500\
#             --target_class=2\
#             --trojan_epochs=500\
#             --dataset=ogbn-arxiv
#     done    
# done



# Rand: Flickr
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_SBA.py \
#             --train_lr 0.02 \
#             --hidden 64 \
#             --prune_thr=0.4\
#             --attack_method=Rand_Samp\
#             --vs_size=200\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=Flickr
#     done    
# done

# Rand: Flickr
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_SBA.py \
#             --train_lr 0.02 \
#             --hidden 64 \
#             --prune_thr=0.4\
#             --attack_method=Rand_Gene\
#             --vs_size=565\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=ogbn-arxiv
#     done    
# done

# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_SBA.py \
#             --train_lr 0.02 \
#             --hidden 64 \
#             --prune_thr=0.4\
#             --attack_method=Rand_Samp\
#             --vs_size=80\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=Flickr
#     done    
# done