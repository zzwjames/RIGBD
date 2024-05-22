# models=(GCN GraphSage GAT)
models=(GCN)
# isolate means the Prune+LD defense method
# defense_modes=(none prune isolate)
defense_modes=(none)
vs_numbers=(40)
# Cora
# for defense_mode in ${defense_modes[@]};
# do 
    # for model in ${models[@]};
    # do
    #     python -u run_adaptive.py \
    #         --prune_thr=0.1\
    #         --dataset=Cora\
    #         --homo_loss_weight=0\
    #         --vs_number=40\
    #         --hidden=64\
    #         --seed=12\
    #         --test_model=${model}\
    #         --defense_mode=none\
    #         --selection_method=none\
    #         --homo_boost_thrd=0.5\
    #         --epochs=200\
    #         --trojan_epochs=400
    # done    
# done

# # Pubmed
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
        python -u run_adaptive.py \
            --prune_thr=0.2\
            --dataset=Pubmed\
            --homo_loss_weight=100\
            --vs_number=160\
            --test_model=GCN\
            --defense_mode=none\
            --selection_method=none\
            --homo_boost_thrd=0.1\
            --epochs=2000\
            --target_class=2\
            --trojan_epochs=400
#     done    
# done

# # Flickr
# for defense_mode in ${defense_modes[@]};
# do 
# for vs_number in ${vs_numbers[@]};
# do
    # python -u run_adaptive.py \
    #     --prune_thr=0.2\
    #     --dataset=Flickr\
    #     --homo_loss_weight=0\
    #     --hidden=128\
    #     --vs_number=160\
    #     --model=GCN\
    #     --test_model=GCN\
    #     --target_class=0\
    #     --seed=15\
    #     --defense_mode=none\
    #     --selection_method=none\
    #     --homo_boost_thrd=0.9\
    #     --trigger_size=3\
    #     --evaluate_mode=overall\
    #     --epochs=801\
    #     --trojan_epochs=801
# done    
# done

# # OGBN-Arixv
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_adaptive.py \
#             --prune_thr=0.8\
#             --dataset=ogbn-arxiv\
#             --homo_loss_weight=200\
#             --vs_number=565\
#             --model=GCN\
#             --hidden=128\
#             --test_model=GCN\
#             --test_model=GraphSage\
#             --defense_mode=none\
#             --selection_method=none\
#             --homo_boost_thrd=0.8\
#             --epochs=800\
#             --target_class=2\
#             --trojan_epochs=800
#     done    
# done
