CUDA_VISIBLE_DEVICES=7 python var_dist_analysis.py --mode var --partial 7500 --dataset imagenet-a
# CUDA_VISIBLE_DEVICES=7 python var_dist_analysis.py --mode l2_dist --partial 500 --dataset imagenet-a
# CUDA_VISIBLE_DEVICES=7 python var_dist_analysis.py --mode l2_dist --partial 500 --plot_dist_kde --extra kde_plot

# for cfg in $(seq 0.0 0.5 8.0); do
#     CUDA_VISIBLE_DEVICES=7 python var_dist_analysis.py --mode l2_dist --partial 500 --cfg $cfg
# done

# for k in $(seq 50 50 500); do
#     CUDA_VISIBLE_DEVICES=7 python var_dist_analysis.py --mode l2_dist --partial 500 --top_k $k
# done

