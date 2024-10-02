#!/bin/bash

#fake

python process_to_image.py --seed 42 --dataset UCF-101 --video_type real_1 --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --batch 4
python process_to_image.py --seed 42 --dataset UCF-101 --video_type real_2 --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --batch 4
python process_to_image.py --seed 42 --dataset UCF-101 --video_type videocrafter_freenoise --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --batch 4
python process_to_image.py --seed 42 --dataset UCF-101 --video_type videocrafter_fifo --video_resolution 256 --num_frames 128 --num_clip_samples 2048 --batch 4


# python compute_fid.py --dataset kinetics400

echo "모든 작업이 완료되었습니다."
