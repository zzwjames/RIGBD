# models=(GCN GraphSage GAT)
# # isolate means the Prune+LD defense method
# defense_modes=(none prune isolate)

# # Cora
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_adaptive.py \
#             --prune_thr=0.1\
#             --dataset=Cora\
#             --homo_loss_weight=50\
#             --vs_number=10\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --selection_method=cluster_degree\
#             --homo_boost_thrd=0.5\
#             --epochs=200\
#             --trojan_epochs=400
#     done    
# done

models=(GCN)
# isolate means the Prune+LD defense method
defense_modes=(reconstruct)
seeds=(12 25)
# target_classes=(6 0 1 2 3 4 5 6 7 8 9 10 11 21 12 13 14 15 16 17 18 19 20 21 22 23 24 28 29 30 31 32 33 34 35 36 37 38 39 40)
# target_classes=(6)
# Cora
for defense_mode in ${defense_modes[@]};
do 
    for model in ${models[@]};
    do
        for seed in ${seeds[@]}:
        do
            python -u run_adaptive.py \
                --prune_thr=0.1\
                --dataset=Cora\
                --trigger_size=3\
                --homo_loss_weight=50\
                --vs_number=10\
                --target_class=1\
                --seed=10\
                --test_model=${model}\
                --defense_mode=${defense_mode}\
                --selection_method=none\
                --homo_boost_thrd=0.5\
                --epochs=200\
                --trojan_epochs=101\
                --hidden=32
        done
    done    
done

# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_adaptive.py \
#             --prune_thr=0.1\
#             --dataset=Citeseer\
#             --homo_loss_weight=50\
#             --vs_number=30\
#             --hidden 256\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --selection_method=none\
#             --homo_boost_thrd=0.85\
#             --trigger_size=1\
#             --weight_target=1\
#             --weight_ood=1\
#             --target_class=2\
#             --weight_target_class=1\
#             --epochs=501\
#             --k=200\
#             --rec_epochs=10\
#             --trojan_epochs=501
#     done    
# done

# Pubmed
# trigger_size=3
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         for seed in ${seeds[@]}:
#         do 
#             python -u defense.py \
#                 --prune_thr=0.2\
#                 --dataset=Pubmed\
#                 --homo_loss_weight=100\
#                 --hidden=256\
#                 --vs_number=40\
#                 --target_class=2\
#                 --seed=${seed}\
#                 --test_model=${model}\
#                 --defense_mode=${defense_mode}\
#                 --selection_method=none\
#                 --homo_boost_thrd=0.9\
#                 --weight_target_class=1\
#                 --weight_target=1\
#                 --weight_ood=1\
#                 --outter_size=4096\
#                 --trigger_size=3\
#                 --weight_target_class=1\
#                 --epochs=501\
#                 --trojan_epochs=501
#         done
#     done    
# done

# # Flickr
## hidden=256
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do  
#         for seed in ${seeds[@]};
#         do 
#             python -u run_adaptive.py \
#                 --prune_thr=0.4\
#                 --dataset=Flickr\
#                 --hidden=32\
#                 --homo_loss_weight=160\
#                 --vs_number=80\
#                 --test_model=${model}\
#                 --defense_mode=${defense_mode}\
#                 --selection_method=none\
#                 --homo_boost_thrd=0.5\
#                 --target_class=6\
#                 --weight_ood=1\
#                 --trigger_size=3\
#                 --range=0.1\
#                 --epochs=801\
#                 --trojan_epochs=801
#         done
#     done    
# done

# OGBN-Arixv
# train: vs:500
# test: vs:1500
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         for target_class in ${target_classes[@]};
#         do
#             for seed in ${seeds[@]};
#             do
#                 python -u run_adaptive.py \
#                     --prune_thr=0.8\
#                     --dataset=ogbn-arxiv\
#                     --homo_loss_weight=500\
#                     --vs_number=565\
#                     --hidden=256\
#                     --test_model=${model}\
#                     --defense_mode=${defense_mode}\
#                     --selection_method=none\
#                     --homo_boost_thrd=0.95\
#                     --weight_target_class=1\
#                     --weight_target=1\
#                     --epochs=2000\
#                     --seed=${seed}\
#                     --trigger_size=3\
#                     --weight_ood=1\
#                     --target_class=${target_class}\
#                     --trojan_epochs=2001
#             done
#         done
#     done    
# done