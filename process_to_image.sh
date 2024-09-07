#!/bin/bash

#fake


python process_to_image.py --seed 42 --dataset kinetics400 --video_type real_1 --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type real_2 --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type repeated_patterns_mix --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type repeated_patterns_one --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type smooth_transitions --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .0 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .3 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity .5 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed --severity 1. --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental --severity 1. --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental --severity .6 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .0 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .3 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity .5 --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_fixed_temporal --severity 1. --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental_temporal --severity 1. --batch 4
python process_to_image.py --seed 42 --dataset kinetics400 --video_type fake --video_resolution 128 --num_frames 128 --num_clip_samples 2048 --distortion_type mix_incremental_temporal --severity .6 --batch 4

# python compute_fid.py --dataset kinetics400

echo "모든 작업이 완료되었습니다."
