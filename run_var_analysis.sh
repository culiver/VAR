# CUDA_VISIBLE_DEVICES=7 python var_analysis.py --mode var --partial 7500 --dataset imagenet-a
CUDA_VISIBLE_DEVICES=7 python var_analysis.py --mode var --partial 7500 --dataset imagenet-a --depth 30

# CUDA_VISIBLE_DEVICES=7 python analysis.py --partial 500 --cfg 0.5
# for cfg in $(seq 0.0 0.5 8.0); do
#     CUDA_VISIBLE_DEVICES=6 python var_dist_analysis.py --mode var --partial 500 --cfg $cfg
#     # CUDA_VISIBLE_DEVICES=7 python analysis.py --partial 500 --cfg $cfg
# done


