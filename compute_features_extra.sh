#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .2 --batch 4 
CUDA_VISIBLE_DEVICES=7 python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .2 --batch 4 
CUDA_VISIBLE_DEVICES=7 python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .2 --batch 4 --no_resize