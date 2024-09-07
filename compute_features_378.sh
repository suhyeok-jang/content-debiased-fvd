#!/bin/bash

# CUDA 장치 리스트 (GPU 4, 5번 사용)
cuda_devices=(1 2 3 4 5 6 7)

tasks=(
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type real_1 --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type real_2 --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type repeated_patterns_mix --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type repeated_patterns_one --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type smooth_transitions --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .0 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .3 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .5 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity 1. --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental --severity 1. --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental --severity .6 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .0 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .3 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .5 --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity 1. --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental_temporal --severity 1. --batch 1 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 378 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental_temporal --severity .6 --batch 1 --no_resize"
)

# 총 작업 수, 17
num_tasks=${#tasks[@]}

# 첫 두 번의 순회에서는 GPU 1~7번에 작업을 균등하게 할당
for ((i=0; i<num_tasks-3; i+=7)); do
    for j in {0..6}; do  # GPU 1~7에 할당
        task_idx=$((i + j))
        if [ $task_idx -lt $num_tasks ]; then
            CUDA_VISIBLE_DEVICES=${cuda_devices[$j]} ${tasks[$task_idx]} &
        fi
    done
    wait  # 현재 작업 세트가 끝날 때까지 대기
done

# 마지막 3개의 작업을 GPU 5, 6, 7에 할당
for ((i=num_tasks-3; i<num_tasks; i++)); do
    gpu_idx=$((i - 17 + 7))  # GPU 5,6,7에 할당
    CUDA_VISIBLE_DEVICES=$gpu_idx ${tasks[$i]} &
done

wait  # 마지막 세트의 작업이 끝날 때까지 대기

echo "모든 작업이 완료되었습니다."