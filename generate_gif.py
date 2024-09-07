import warnings
warnings.filterwarnings('ignore')
import imageio
import os
import argparse
import torch.nn as nn

from cdfvd import fvd

def video_to_gif(video, gif_path='output.gif'):
    # video: CTHW format
    C, T, H, W = video.shape
    frames = []

    for t in range(T):
        frame = video[:, t, :, :]  # Get the t-th frame (CHW format)
        frame = (frame*255).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for visualization
        frames.append(frame.astype('uint8'))

    # Save as GIF
    imageio.mimsave(gif_path, frames, duration = 0.01)
    print(f'GIF saved to {gif_path}')
    
def main(args):  
    if args.dataset == "sky_timelapse": #2048 clip이 나오지 않음
        args.num_clip_samples = 'full'  
        
    evaluator = fvd.cdfvd('videomae', n_real = args.num_clip_samples, n_fake = args.num_clip_samples, seed=args.seed, compute_feats =False, device = "cuda", half_precision = True)
    
    real_video, real_video_loader = evaluator.load_videos(video_info= f'/data/sky_timelapse/sky_timelapse/sky_train/', 
        data_type='image_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
        num_workers =4, batch_size=args.batch)
    
    gif_path = f'/home/jsh/content-debiased-fvd/examples/{args.dataset}/real/ref.gif'
    if not os.path.exists(os.path.dirname(gif_path)):
        os.makedirs(os.path.dirname(gif_path))
    video_to_gif(real_video, gif_path= gif_path)
       
    # for distortion_type in ['s','st','half_s_st','last_16_st','per_8_st']:
    #     for distortion_name in ['elastic_transform','motion_blur']:
    #         for severity in [1, 3, 5]:
                
    #             if args.dataset == "sky_timelapse":
    #                 fake_video, fake_video_loader = evaluator.load_videos(video_info= f'/data/sky_timelapse/sky_timelapse/sky_train/', 
    #                                         data_type='image_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
    #                                         corrupt = distortion_type, corrupt_name = distortion_name, corrupt_severity= severity, num_workers =4, batch_size=args.batch)
                
    #             else:
    #                 fake_video, fake_video_loader = evaluator.load_videos(video_info= f'/data/{args.dataset}/', 
    #                             data_type='video_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
    #                             corrupt = distortion_type, corrupt_name = distortion_name, corrupt_severity= severity, num_workers =4, batch_size=args.batch)
                
    #             gif_path = f'/home/jsh/content-debiased-fvd/examples/{args.dataset}/fake/{distortion_type}/{distortion_name}/severity_{severity}.gif'
                
    #             if not os.path.exists(os.path.dirname(gif_path)):
    #                 os.makedirs(os.path.dirname(gif_path))
    #             video_to_gif(fake_video, gif_path= gif_path)
                
    # gif_path = f'/home/jsh/content-debiased-fvd/examples/{args.dataset}/fake/{args.distortion_type}/{args.distortion_name}/severity_{args.severity}.gif'
    # if not os.path.exists(os.path.dirname(gif_path)):
    #     os.makedirs(os.path.dirname(gif_path))
    # video_to_gif(fake_video, gif_path= gif_path)
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the latent representation of video clips by VideoMAE')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", type=str, choices=['UCF-101', 'sky_timelapse'])
    parser.add_argument("--video_type", type=str, choices=['real','fake'])
    parser.add_argument("--video_resolution", default=128, type=int)
    parser.add_argument("--num_frames", default=128, type=int)
    parser.add_argument("--num_clip_samples", default=2048, type=int)
    parser.add_argument("--distortion_type", type=str, choices=['s','st','half_s_st','last_16_st','per_8_st'])
    parser.add_argument("--distortion_name", type=str, choices=['elastic_transform','motion_blur'])
    parser.add_argument("--severity", type=int, choices=[1,2,3,4,5])
    parser.add_argument("--batch", default=1, type=int)
    
    args = parser.parse_args()
    
    main(args)
          
             
# video_to_gif(s_video, gif_path=f'/home/jsh/content-debiased-fvd/examples/{distortion}_s_video_severity_{i}.gif')
# video_to_gif(st_video, gif_path=f'/home/jsh/content-debiased-fvd/examples/{distortion}_st_video_severity_{i}.gif')
        

# evaluator.compute_fake_stats(evaluator.load_videos('./example_videos/', data_type='video_folder'))
# score = evaluator.compute_fvd_from_stats()