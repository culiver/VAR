# CUDA_VISIBLE_DEVICES=7 python analysis.py --partial 500 --cfg 0.5
for cfg in $(seq 0.5 0.5 8.0); do
    CUDA_VISIBLE_DEVICES=7 python analysis.py --partial 500 --cfg $cfg
done


