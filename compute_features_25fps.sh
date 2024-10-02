#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python compute_features.py --seed 42 --dataset UCF-101 --video_type real_1 --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --batch 4 --cut_front 16 &
CUDA_VISIBLE_DEVICES=1 python compute_features.py --seed 42 --dataset UCF-101 --video_type real_2 --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --batch 4 --cut_front 16 &
CUDA_VISIBLE_DEVICES=2 python compute_features.py --seed 42 --dataset UCF-101 --video_type videocrafter_freenoise --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --batch 4 --cut_front 16 &
CUDA_VISIBLE_DEVICES=3 python compute_features.py --seed 42 --dataset UCF-101 --video_type videocrafter_fifo --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --batch 4 --cut_front 16 &