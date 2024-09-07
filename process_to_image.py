import warnings
warnings.filterwarnings('ignore')
import imageio
import os
import argparse
import torch.nn as nn
from tqdm import tqdm

from torchvision.utils import save_image
from cdfvd.utils.metric_utils import seed_everything

from cdfvd import fvd
from cdfvd.utils.data_utils import DISTORTIONS, PATTERN_CORRUPTIONS, MIX_CORRUPTIONS


def save_frames_from_dataset(dataloader, output_folder, max_samples, seed):
    """
    주어진 VideoDataset에서 프레임을 추출하여 각 프레임을 개별 이미지로 저장하는 함수.
    
    Args:
        dataloader: DataLoader 인스턴스
        output_folder: 프레임을 저장할 폴더 경로
        max_samples: 최대 저장할 샘플 수
        seed: 랜덤 시드
    """
    # 랜덤 시드 설정
    seed_everything(seed)
    os.makedirs(output_folder, exist_ok=True)

    sample_count = 0
    for batch_idx, data in enumerate(tqdm(dataloader, desc="Processing videos")):
        if sample_count >= max_samples:
            break

        videos = data['video']  # 'video' 키에서 비디오 데이터 추출 -> BCTHW
        B, C, T, H, W = videos.shape

        # videos를 C B T H W -> C (B*T) H W 형식으로 변환
        videos = videos.permute(1, 0, 2, 3, 4).contiguous().view(C, -1, H, W)

        # 이미지 저장
        for frame_idx in range(videos.size(1)):  # B*T개의 프레임
            frame = videos[:, frame_idx, :, :]  # CBTHW -> CTHW
            frame_filename = os.path.join(output_folder, f"batch_{batch_idx}_frame_{frame_idx}.png")
            save_image(frame, frame_filename)

        sample_count += B

def main(args): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
       
    #여기서 videomae는 그냥 설정이므로 상관없음 -> model을 GPU에 띄우는 것이 아님
    evaluator = fvd.cdfvd(None, n_real = args.num_clip_samples, n_fake = args.num_clip_samples, seed=args.seed, compute_feats =False, device = "cpu", half_precision = True)
    
    if args.video_type == 'real_1':
        real_video_loader = evaluator.load_videos(video_info= f'/data/{args.dataset}/valid_subset1', 
                                    data_type='video_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
                                    num_workers =4, batch_size=args.batch) #num_workers를 0으로 조절하면 error가 안남 / 0: multi-processing 사용 X
        
        output_folder = f'./{args.dataset}/cliptoimages/real_1/'
        
        save_frames_from_dataset(real_video_loader, output_folder, args.num_clip_samples, args.seed)
        

    elif args.video_type == 'real_2':
        real_video_loader = evaluator.load_videos(video_info= f'/data/{args.dataset}/valid_subset2', 
                                    data_type='video_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
                                    num_workers =4, batch_size=args.batch) 

        output_folder = f'./{args.dataset}/cliptoimages/real_2/'
    
        save_frames_from_dataset(real_video_loader, output_folder, args.num_clip_samples, args.seed)
    
    else:
        fake_video_loader = evaluator.load_videos(video_info= f'/data/{args.dataset}/valid_subset2', 
                                        data_type='video_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
                                        corrupt = args.distortion_type, corrupt_severity= args.severity, num_workers =4, batch_size=args.batch)
        
        if args.distortion_type in PATTERN_CORRUPTIONS:
            output_folder = f'./{args.dataset}/cliptoimages/fake/{args.distortion_type}/'
        else:
            output_folder = f'./{args.dataset}/cliptoimages/fake/{args.distortion_type}/{args.severity}/'

        save_frames_from_dataset(fake_video_loader, output_folder, args.num_clip_samples, args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the latent representation of video clips by VideoMAE')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", type=str, choices=['UCF-101', 'sky_timelapse','kinetics400'])
    parser.add_argument("--video_type", type=str, choices=['real_1','real_2', 'fake'])
    parser.add_argument("--video_resolution", default=128, type=int)
    parser.add_argument("--num_frames", default=128, type=int)
    parser.add_argument("--num_clip_samples", default=2048, type=int)
    parser.add_argument("--distortion_type", type=str, default = 'repeated_patterns_mix', choices= DISTORTIONS)
    parser.add_argument("--distortion_name", type=str, default = 'elastic_transform',choices=['elastic_transform','motion_blur'])
    parser.add_argument("--severity", type= float, default = 1.)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--fvd_16", action = "store_true")
    
    args = parser.parse_args()
    
    main(args)
          
             
# video_to_gif(s_video, gif_path=f'/home/jsh/content-debiased-fvd/examples/{distortion}_s_video_severity_{i}.gif')
# video_to_gif(st_video, gif_path=f'/home/jsh/content-debiased-fvd/examples/{distortion}_st_video_severity_{i}.gif')
        

# evaluator.compute_fake_stats(evaluator.load_videos('./example_videos/', data_type='video_folder'))
# score = evaluator.compute_fvd_from_stats()