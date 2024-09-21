#!/bin/bash

#fake
# CUDA_VISIBLE_DEVICES=6 python process_to_image.py --seed 42 --dataset UCF-101 --video_type videocrafter_fifo --video_resolution 256 --num_frames 128 --sample_every_n_frames 1 --num_clip_samples 2048 --batch 4 --cut_front 64
CUDA_VISIBLE_DEVICES=7 python process_to_image.py --seed 42 --dataset UCF-101 --video_type videocrafter_fifo --video_resolution 256 --num_frames 128 --sample_every_n_frames 1 --num_clip_samples 2048 --batch 4 --cut_front 16

# python compute_fid.py --dataset kinetics400

# 모든 백그라운드 작업이 완료될 때까지 대기
wait
echo "모든 작업이 완료되었습니다."
