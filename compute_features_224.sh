#!/bin/bash

# CUDA 장치 리스트 (GPU 4, 5번 사용)
cuda_devices=(4 5 6 7)

tasks=(
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type real_1 --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type real_2 --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type repeated_patterns_mix --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type repeated_patterns_one --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type smooth_transitions --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .0 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .3 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .5 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity 1. --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental --severity 1. --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental --severity .6 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .0 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .3 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .5 --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity 1. --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental_temporal --severity 1. --batch 4 --no_resize"
    "python compute_features.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 224 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental_temporal --severity .6 --batch 4 --no_resize"
)

# 총 작업 수
num_tasks=${#tasks[@]}

# 작업을 4개씩 실행, GPU 4, 5, 6, 7번을 순회하면서 사용
for ((i=0; i<num_tasks; i+=4)); do
    for j in {0..3}; do  # 4개의 GPU (0 -> 4번, 1 -> 5번, 2 -> 6번, 3 -> 7번)
        task_idx=$((i + j))
        if [ $task_idx -lt $num_tasks ]; then
            CUDA_VISIBLE_DEVICES=${cuda_devices[$j]} ${tasks[$task_idx]} &
        fi
    done
    wait  # 현재 작업 세트가 끝날 때까지 대기
done

echo "모든 작업이 완료되었습니다."