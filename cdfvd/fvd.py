import os
import scipy
import torch
import numpy as np
import imageio
from .utils.data_utils import get_dataloader, VID_EXTENSIONS, DISTORTIONS, PATTERN_CORRUPTIONS, MIX_CORRUPTIONS
from .utils.metric_utils import seed_everything, FeatureStats

import requests
from tqdm import tqdm
from einops import rearrange

from .third_party.VideoMAEv2.utils import load_videomae_model, preprocess_videomae
from .third_party.i3d.utils import load_i3d_model, preprocess_i3d

from typing import List, Optional, Union
import numpy.typing as npt
import torch.nn as nn

def video_to_gif(video, gif_path):
    # video: CTHW format
    C, T, H, W = video.shape
    frames = []

    for t in range(T):
        frame = video[:, t, :, :]  # Get the t-th frame (CHW format)
        frame = (frame*255).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for visualization
        frames.append(frame.astype('uint8'))

    if not os.path.exists(os.path.dirname(gif_path)):
        os.makedirs(os.path.dirname(gif_path))

    imageio.mimsave(gif_path, frames, duration = int(1000/25)) #25FPS
    print(f'GIF saved to {gif_path}')

def get_videomae_features(stats, model, videos, batchsize=16, device='cuda', model_dtype=torch.float32, concat = False, resize=True):
    vid_length = videos.shape[0] #og : BTHWC / concat: 8B
    for i in range(0, videos.shape[0], batchsize): # (0,1,16) 이어도 오류는 안남
        batch = videos[i:min(vid_length, i + batchsize)]
        input_data = preprocess_videomae(batch, resize)  # torch.Size([B, 3, T, H, W])
        input_data = input_data.to(device=device, dtype=model_dtype, non_blocking = True)
        with torch.no_grad():
            features = model.forward_features(input_data)  # forward_features 호출
            if concat == True:
                features = features.view(vid_length//8, 8, -1) #8개 분리 -> [1,8,1408]
                features = features.view(vid_length//8, -1) #concat -> [1, 11264]
            stats.append_torch(features, num_gpus=1, rank=0)
    return stats


def get_i3d_logits(stats, i3d, videos, batchsize=16, device='cuda', model_dtype=torch.float32):
    vid_length = videos.shape[0]
    for i in range(0, vid_length, batchsize):
        batch = videos[i:min(vid_length, i + batchsize)]
        input_data = preprocess_i3d(batch)
        input_data = input_data.to(device=device, dtype=model_dtype)
        with torch.no_grad():
            features = i3d(input_data)
            stats.append_torch(features, num_gpus=1, rank=0)
    return stats


class cdfvd(object):
    '''This class loads a pretrained model (I3D or VideoMAE) and contains functions to compute the FVD score between real and fake videos.

    Args:
        model: Name of the model to use, either `videomae` or `i3d`.
        n_real: Number of real videos to use for computing the FVD score, if `'full'`, all the videos in the dataset will be used.
        n_fake: Number of fake videos to use for computing the FVD score, if `'full'`, all the videos in the dataset will be used.
        ckpt_path: Path to save the model checkpoint.
        seed: Random seed.
        compute_feats: Whether to compute all features or just mean and covariance.
        device: Device to use for computing the features.
        half_precision: Whether to use half precision for the model.
    '''
    def __init__(self, model: str = 'i3d', dataset: str = "kinetics400", n_real: str = 'full', n_fake: int = 2048, ckpt_path: Optional[str] = None,
                 seed: int = 42, compute_feats: bool = False, device: str = 'cuda', half_precision: bool = False,
                 *args, **kwargs):
        self.model_name = model
        self.ckpt_path = ckpt_path
        self.seed = seed
        self.device = device
        self.n_real = n_real
        self.n_fake = n_fake
        #capture all -> Gaussian에 필요한 mean, cov말고도 모든 feature 계산
        self.real_stats = FeatureStats(max_items=None if n_real == 'full' else n_real, capture_mean_cov=True, capture_all=compute_feats)
        self.fake_stats = FeatureStats(max_items=None if n_fake == 'full' else n_fake, capture_mean_cov=True, capture_all=compute_feats)
        self.model_dtype = (
            torch.float16 if half_precision else torch.float32
        )
        if self.model_name is not None:
            assert self.model_name in ['videomae', 'i3d']
            print('Loading %s model ...' % self.model_name)
            if self.model_name == 'videomae':
                #eval mode로 진입 (layer 동작 inference에 맞게 change)
                self.model = load_videomae_model(torch.device(device), ckpt_path).eval().to(dtype=self.model_dtype)
                self.model.head.cpu()
                torch.cuda.empty_cache()
                #with torch.no_grad() (off the autograd engine)
                self.feature_fn = get_videomae_features
            else:
                self.model = load_i3d_model(torch.device(device), ckpt_path).eval().to(dtype=self.model_dtype)
                self.feature_fn = get_i3d_logits
                
        self.dataset = dataset

    def compute_fvd_from_stats(self, fake_stats: Optional[FeatureStats] = None, real_stats: Optional[FeatureStats] = None) -> float:
        '''This function computes the FVD score between real and fake videos using precomputed features.
        If the stats are not provided, it uses the stats stored in the object.
        
        Args:
            fake_stats: `FeatureStats` object containing the features of the fake videos.
            real_stats: `FeatureStats` object containing the features of the real videos.
        
        Returns:
            FVD score between the real and fake videos.
        '''
        fake_stats = self.fake_stats if fake_stats is None else fake_stats
        real_stats = self.real_stats if real_stats is None else real_stats
        mu_fake, sigma_fake = fake_stats.get_mean_cov()
        mu_real, sigma_real = real_stats.get_mean_cov()
        m = np.square(mu_real - mu_fake).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_real, sigma_fake), disp=False)
        return np.real(m + np.trace(sigma_fake + sigma_real - s * 2))
    
    def compute_fvd(self, real_videos: npt.NDArray[np.uint8], fake_videos: npt.NDArray[np.uint8]) -> float:
        '''
        This function computes the FVD score between real and fake videos in the form of numpy arrays.

        Args:
            real_videos: A numpy array of videos with shape `(B, T, H, W, C)`, values in the range `[0, 255]`
            fake_videos: A numpy array of videos with shape `(B, T, H, W, C)`, values in the range `[0, 255]`
        
        Returns:
            FVD score between the real and fake videos.
        '''
        self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype)
        self.fake_stats = self.feature_fn(self.fake_stats, self.model, fake_videos, device=self.device, model_dtype=self.model_dtype)
        return self.compute_fvd_from_stats(
            self.fake_stats, self.real_stats)

    def compute_real_stats(self, loader: Union[torch.utils.data.DataLoader, List, None] = None, concat = False, make_gif = False, resize= True, video_type = 'real_1', cut_front = 0) -> FeatureStats:
        '''
        This function computes the real features from a dataset.

        Args:
            loader: real videos, either in the type of dataloader or list of numpy arrays.

        Returns:
            FeatureStats object containing the features of the real videos.
        '''
        #Dataloader에서 데이터가 섞이는 과정은 첫 번째 데이터 로드 시점에 발생하므로
        #Dataloader를 만들고 나서 seed를 설정해도 고정된 비디오, 이미지가 나오게 됨
        seed_everything(self.seed)
        if loader is None:
            assert self.real_stats.max_items is not None
            return

        self.resolution = loader.dataset.resolution
        self.frames = loader.dataset.sequence_length
        self.sample_every_n_frames = loader.dataset.sample_every_n_frames
        self.actual_frames = self.frames // self.sample_every_n_frames
        
        while self.real_stats.max_items is None or self.real_stats.num_items < self.real_stats.max_items:
            for i, batch in enumerate(tqdm(loader)):
                if make_gif and i<=30:
                    if cut_front != 0:
                        gif_path = f'/home/jsh/content-debiased-fvd/examples/{self.dataset}/{self.resolution}/{cut_front}frames(x{self.sample_every_n_frames})/{video_type}/{batch['label'][0]}/{i}.gif'
                        video_to_gif(batch['video'][:,:,:cut_front,:,:][0], gif_path= gif_path) #CTHW
                    else:
                        gif_path = f'/home/jsh/content-debiased-fvd/examples/{self.dataset}/{self.resolution}/{self.actual_frames}frames(x{self.sample_every_n_frames})/{video_type}/{batch['label'][0]}/{i}.gif'
                        video_to_gif(batch['video'][0], gif_path= gif_path) #CTHW

                if concat == False:
                    if cut_front != 0 :
                        batch['video'] = batch['video'][:,:,:cut_front,:,:]
                    real_videos = rearrange(batch['video']*255, 'b c t h w -> b t h w c').byte().data.numpy()
                else:
                    b, c, t, h, w = batch['video'].shape
                    real_videos = rearrange(batch['video']*255, f'b c t h w -> b t h w c')
                    real_videos = rearrange(real_videos, f'b (t1 t2) h w c -> (b t1) t2 h w c', t1=8).byte().data.numpy()
                self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype, concat = concat, resize= resize)
                if self.real_stats.max_items is not None and self.real_stats.num_items >= self.real_stats.max_items:
                    break
            if self.real_stats.max_items is None:
                break

        return self.real_stats
    
    def compute_fake_stats(self, loader: Union[torch.utils.data.DataLoader, List, None] = None, concat = False, make_gif = False, resize = True, video_type = "fake", cut_front = 0) -> FeatureStats:
        '''
        This function computes the fake features from a dataset.
        
        Args:
            loader: fake videos, either in the type of dataloader or list of numpy arrays.
        
        Returns:
            FeatureStats object containing the features of the fake videos.
        '''
        seed_everything(self.seed)
        
        self.distortion_type = loader.dataset.corrupt
        self.severity = loader.dataset.corrupt_severity
        self.resolution = loader.dataset.resolution
        self.frames = loader.dataset.sequence_length
        
        while self.fake_stats.max_items is None or self.fake_stats.num_items < self.fake_stats.max_items:
            for i,batch in enumerate(tqdm(loader)):
                if make_gif and i<=30:
                    if self.distortion_type in MIX_CORRUPTIONS:
                        gif_path = f'/home/jsh/content-debiased-fvd/examples/{self.dataset}/{self.resolution}/{self.frames}frames/{video_type}/{batch['label'][0]}/{i}.gif'
                    else:
                        if cut_front != 0:
                            gif_path = f'/home/jsh/content-debiased-fvd/examples/{self.dataset}/{self.resolution}/{cut_front}frames/{video_type}/{batch['label'][0]}/{i}.gif'
                        else:
                            gif_path = f'/home/jsh/content-debiased-fvd/examples/{self.dataset}/{self.resolution}/{self.frames}frames/{video_type}/{batch['label'][0]}/{i}.gif'
                    
                    if cut_front != 0:
                        video_to_gif(batch['video'][:,:,:cut_front,:,:][0], gif_path= gif_path) #CTHW
                    else:
                        video_to_gif(batch['video'][0], gif_path= gif_path) #CTHW
                    
                if concat == False:
                    if cut_front != 0:
                        batch['video'] = batch['video'][:,:,:cut_front,:,:] #bcthw
                        assert batch['video'].shape[2] == cut_front, f"Error: Expected 16 frames, but got {batch['video'].shape[2]} frames."
                        
                    fake_videos = rearrange(batch['video']*255, 'b c t h w -> b t h w c').byte().data.numpy()
                else:
                    b, c, t, h, w = batch['video'].shape
                    fake_videos = rearrange(batch['video']*255, f'b c t h w -> b t h w c')
                    fake_videos = rearrange(fake_videos, f'b (t1 t2) h w c -> (b t1) t2 h w c', t1=8).byte().data.numpy()
                self.fake_stats = self.feature_fn(self.fake_stats, self.model, fake_videos, device=self.device, model_dtype=self.model_dtype, concat=concat, resize = resize)
                # n_real clip개수로 제한 (2048)
                if self.fake_stats.max_items is not None and self.fake_stats.num_items >= self.fake_stats.max_items:
                    break
            if self.fake_stats.max_items is None:
                break

        return self.fake_stats


    def add_real_stats(self, real_videos: npt.NDArray[np.uint8]):
        '''
        This function adds features of real videos to the real_stats object.

        Args:
            real_videos: A numpy array of videos with shape `(B, T, H, W, C)`, values in the range `[0, 255]`.
        '''
        self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype)

    def add_fake_stats(self, fake_videos: npt.NDArray[np.uint8]):
        '''
        This function adds features of fake videos to the fake_stats object.
        
        Args:
            fake_videos: A numpy array of videos with shape `(B, T, H, W, C)`, values in the range `[0, 255]`.
        '''
        self.fake_stats = self.feature_fn(self.fake_stats, self.model, fake_videos, device=self.device, model_dtype=self.model_dtype)
    
    def empty_real_stats(self):
        '''
        This function empties the real_stats object.
        '''
        self.real_stats = FeatureStats(max_items=self.real_stats.max_items, capture_mean_cov=True)

    def empty_fake_stats(self):
        '''
        This function empties the fake_stats object.
        '''
        self.fake_stats = FeatureStats(max_items=self.fake_stats.max_items, capture_mean_cov=True)
    
    def save_real_stats(self, path: str):
        '''
        This function saves the real_stats object to a file.

        Args:
            path: Path to save the real_stats object.
        '''
        self.real_stats.save(path)
        print('Real stats saved to %s' % path)
    
    def save_fake_stats(self, path: str):
        '''
        This function saves the fake_stats object to a file.

        Args:
            path: Path to save the fake_stats object.
        '''
        self.fake_stats.save(path)
        print('Fake stats saved to %s' % path)
    
    def load_real_stats(self, path: str):
        '''
        This function loads the real_stats object from a file.

        Args:
            path: Path to load the real_stats object.
        '''
        self.real_stats = self.real_stats.load(path)
        print('Real stats loaded from %s' % path)
        
    def load_fake_stats(self, path: str):
        '''
        This function loads the fake_stats object from a file.

        Args:
            path: Path to load the fake_stats object.
        '''
        self.fake_stats = self.fake_stats.load(path)
        print('Fake stats loaded from %s' % path)

    def load_videos(self, video_info: str, resolution: int = 256, sequence_length: int = 16, sample_every_n_frames: int = 1,
                    data_type: str = 'video_numpy', num_workers: int = 4, batch_size: int = 16,
                    corrupt = None, corrupt_name = None, corrupt_severity = None) -> Union[torch.utils.data.DataLoader, List, None]:
        '''
        This function loads videos from a way specified by `data_type`. 
        `video_numpy` loads videos from a file containing a numpy array with the shape `(B, T, H, W, C)`.
        `video_folder` loads videos from a folder containing video files.
        `image_folder` loads videos from a folder containing image files.
        `stats_pkl` indicates that `video_info` of a dataset name for pre-computed features. Currently supports `ucf101`, `kinetics`, `sky`, `ffs`, and `taichi`.

        Args:
            video_info: Path to the video file or folder.
            resolution: Resolution of the video.
            sequence_length: Length of the video sequence.
            sample_every_n_frames: Number of frames to skip.
            data_type: Type of the video data, either `video_numpy`, `video_folder`, `image_folder`, or `stats_pkl`.
            num_workers: Number of workers for the dataloader.
            batch_size: Batch size for the dataloader.
        
        Returns:
            Dataloader or list of numpy arrays containing the videos.
        '''
        if data_type=='video_numpy' or video_info.endswith('.npy'):
            video_array = np.load(video_info)
            video_loader = [{'video':  rearrange(torch.from_numpy(video_array[i:i+batch_size])/255., 'b t h w c -> b c t h w')} for i in range(0, video_array.shape[0], batch_size)]
        elif data_type=='video_folder':
            print('Loading from video files ...')
            video_loader = get_dataloader(video_info, image_folder=False,
                                    resolution=resolution, sequence_length=sequence_length,
                                    sample_every_n_frames=sample_every_n_frames,
                                    batch_size=batch_size, num_workers=num_workers,
                                    corrupt=corrupt, corrupt_name = corrupt_name, corrupt_severity = corrupt_severity)
           
            print(f'{len(video_loader.dataset)}개의 clip이 준비되어 있습니다')
            
        elif data_type=='image_folder':
            print('Loading from frame files ...')
            video_loader = get_dataloader(video_info, image_folder=True,
                                    resolution=resolution, sequence_length=sequence_length,
                                    sample_every_n_frames=sample_every_n_frames,
                                    batch_size=batch_size, num_workers=num_workers,
                                    corrupt=corrupt, corrupt_name = corrupt_name, corrupt_severity = corrupt_severity)
            
            print(f'{len(video_loader.dataset)}개의 clip이 준비되어 있습니다')
            
        elif data_type=='stats_pkl':
            video_loader = None
            cache_name = '%s_%s_%s_res%d_len%d_skip%d_seed%d.pkl' % (self.model_name.lower(), video_info, self.n_real, resolution, sequence_length, sample_every_n_frames, 0)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ckpt_path = os.path.join(current_dir, 'fvd_stats_cache', cache_name)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

            if not os.path.exists(ckpt_path):
                # download the ckpt to the path
                ckpt_url = 'https://content-debiased-fvd.github.io/files/%s' % cache_name
                response = requests.get(ckpt_url, stream=True, allow_redirects=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024

                with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                    with open(ckpt_path, "wb") as fw:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            fw.write(data)

            self.real_stats = self.real_stats.load(ckpt_path)

        else:
            raise ValueError('Invalid real_video path')
        # return video_loader
        return video_loader

    def offload_model_to_cpu(model:nn.Module):
        '''
        This function offloads the model to the CPU to release the memory.
        '''
        model = model.cpu()
        torch.cuda.empty_cache()
