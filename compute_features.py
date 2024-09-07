import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import torch.nn as nn

from cdfvd import fvd
from cdfvd.utils.data_utils import DISTORTIONS, PATTERN_CORRUPTIONS, MIX_CORRUPTIONS
from cdfvd.utils.metric_utils import seed_everything

def main(args):    
    if args.dataset in ("sky_timelapse"): #2048 clip이 나오지 않음 -> 다시봐야함 ㅈㅅ
        args.num_clip_samples = 'full'
        
    evaluator = fvd.cdfvd('videomae', dataset = args.dataset, n_real = args.num_clip_samples, n_fake = args.num_clip_samples, seed=args.seed, compute_feats =False, device = "cuda", half_precision = True)
    # evaluator = fvd.cdfvd('videomae', dataset = args.dataset, n_real = 5, n_fake = 5, seed=args.seed, compute_feats =False, device = "cuda", half_precision = True)
    
    if args.video_type in ('real_1', 'real_2'):
        if args.fvd_16:
            save_dir= os.path.join('/home/jsh/content-debiased-fvd/stats/new/fvd_16x8', args.dataset, args.video_type, str(args.video_resolution)+'_'+str(args.num_frames)+'_'+str(args.num_clip_samples))
        else:
            save_dir= os.path.join('/home/jsh/content-debiased-fvd/stats/new/fvd_128', args.dataset, args.video_type, str(args.video_resolution)+'_'+str(args.num_frames)+'_'+str(args.num_clip_samples))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            raise 'Feature is already computed'
        
        if args.dataset == "sky_timelapse":
            real_video_loader = evaluator.load_videos(video_info= f'/data/sky_timelapse/sky_timelapse/sky_train/', 
                    data_type='image_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
                    num_workers =4, batch_size=args.batch)
            
        if args.dataset == "kinetics400":
            if args.video_type == "real_1":
                real_video_loader = evaluator.load_videos(video_info= f'/data/{args.dataset}/valid_subset1', 
                                    data_type='video_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
                                    num_workers =4, batch_size=args.batch) #num_workers를 0으로 조절하면 error가 안남
            else: #real_2
                real_video_loader = evaluator.load_videos(video_info= f'/data/{args.dataset}/valid_subset2', 
                                    data_type='video_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
                                    num_workers =4, batch_size=args.batch) 
                
        evaluator.compute_real_stats(real_video_loader, concat = args.fvd_16, make_gif = True, resize= args.no_resize) #evaluator FeatureStats에 저장
        # print(evaluator.real_stats.raw_mean)
        save_path = os.path.join(save_dir, 'videomae_feature.pkl')
        evaluator.save_real_stats(save_path)


    else:
        if args.fvd_16:
            save_dir = os.path.join('/home/jsh/content-debiased-fvd/stats/new/fvd_16x8', args.dataset, args.video_type, str(args.video_resolution)+'_'+str(args.num_frames)+'_'+str(args.num_clip_samples), args.distortion_type)
        else:
            save_dir = os.path.join('/home/jsh/content-debiased-fvd/stats/new/fvd_128', args.dataset, args.video_type, str(args.video_resolution)+'_'+str(args.num_frames)+'_'+str(args.num_clip_samples), args.distortion_type, str(args.severity))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            raise 'Feature is already computed'

        if args.dataset == "sky_timelapse":
            fake_video_loader = evaluator.load_videos(video_info= f'/data/sky_timelapse/sky_timelapse/sky_train/', 
                            data_type='image_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
                            corrupt = args.distortion_type, corrupt_name = args.distortion_name, corrupt_severity= args.severity, num_workers =4, batch_size=args.batch)
        
        if args.dataset == "kinetics400":
            #Seed unfixed
            fake_video_loader = evaluator.load_videos(video_info= f'/data/{args.dataset}/valid_subset2', 
                                    data_type='video_folder', resolution= args.video_resolution, sequence_length= args.num_frames, sample_every_n_frames = 1,
                                    corrupt = args.distortion_type, corrupt_severity= args.severity, num_workers =4, batch_size= args.batch)
            

        evaluator.compute_fake_stats(fake_video_loader, concat = args.fvd_16, make_gif=True, resize = args.no_resize) 
        # print(evaluator.fake_stats.raw_mean)
        save_path = os.path.join(save_dir, 'videomae_feature.pkl')
        evaluator.save_fake_stats(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the latent representation of video clips by VideoMAE')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", type=str, choices=['UCF-101', 'sky_timelapse','kinetics400'])
    parser.add_argument("--video_type", type=str, choices=['real_1','real_2', 'fake'])
    parser.add_argument("--video_resolution", default=128, type=int)
    parser.add_argument("--num_frames", default=128, type=int)
    parser.add_argument("--num_clip_samples", default=2048, type=int)
    parser.add_argument("--distortion_type", type=str, default = 'repeated_patterns_mix', choices = DISTORTIONS)
    parser.add_argument("--distortion_name", type=str, default = 'elastic_transform', choices = ['elastic_transform','motion_blur'])
    parser.add_argument("--severity", type= float, default = 1.)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--fvd_16", action = "store_true")
    parser.add_argument("--no_resize", action = "store_false") #활성화시 Resize = False
    
    args = parser.parse_args()
    
    main(args)
          
             