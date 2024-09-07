import os
import math
import os.path as osp
import random
import pickle
import warnings
import copy
from typing import *
import colorsys

import glob
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips

from corruptions.imagecorruptions import corrupt
from scipy.ndimage.interpolation import map_coordinates
from .metric_utils import seed_everything

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
VID_EXTENSIONS = ['.avi', '.mp4', '.webm', '.mov', '.mkv', '.m4v']

# ['s','st','half_s_st','last_16_st','per_8_st', 'repeated_patterns_mix', 'repeated_patterns_one', 'motion_blur_incremental', 'elastic_transform_incremental', 'mix_incremental', 'mix_fixed', "mix_incremental_temporal", "mix_fixed_temporal", "smooth_transitions"]
DISTORTIONS = ['repeated_patterns_mix', 'repeated_patterns_one', "smooth_transitions",
               'mix_incremental', 'mix_fixed', "mix_incremental_temporal", "mix_fixed_temporal"]

PATTERN_CORRUPTIONS = ['repeated_patterns_mix', 'repeated_patterns_one', "smooth_transitions"]
MIX_CORRUPTIONS = ['mix_incremental', 'mix_fixed', "mix_incremental_temporal", "mix_fixed_temporal"]


def get_dataloader(data_path, image_folder, resolution=128, sequence_length=16, sample_every_n_frames=1,
                   batch_size=16, num_workers=8, corrupt = None, corrupt_name = None, corrupt_severity = None):
    data = VideoData(data_path, image_folder, resolution, sequence_length, sample_every_n_frames, 
                     batch_size, num_workers, corrupt = corrupt, corrupt_name = corrupt_name, corrupt_severity = corrupt_severity)
    dataloader = data._dataloader()
    return dataloader


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


def preprocess(video, resolution, sequence_length=None, in_channels=3, sample_every_n_frames=1, corrupt= None, corrupt_name=None, corrupt_severity=None):
    # video: THWC, {0, ..., 255}
    assert in_channels == 3
    
    if (corrupt is None) or corrupt in PATTERN_CORRUPTIONS:
        video = video.permute(0, 3, 1, 2).float() / 255.  # TCHW, [0-1]
    else: #corrupt libray expected to intput range: [0,255], uint8
        video = video.permute(0, 3, 1, 2)
    
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # skip frames
    if sample_every_n_frames > 1:
        video = video[::sample_every_n_frames]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False, antialias=True)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    
    if corrupt is not None:
        if corrupt == 's':
            video = distortion(video, corrupt_name, corrupt_severity, seed=42, option = 'naive')
            video = video.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
            
        elif corrupt == 'st':
            video = distortion(video, corrupt_name, corrupt_severity, seed=None, option= 'naive')
            video = video.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
        
        elif corrupt == 'half_s_st':
            video = distortion(video, corrupt_name, corrupt_severity, option = 'half_s_st')
            video = video.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
        
        elif corrupt == 'last_16_st': #마지막 16 frame만 st
            video = distortion(video, corrupt_name, corrupt_severity, option= 'last_16_st')
            video = video.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
        
        elif corrupt == 'per_8_st': # 8 frame마다 st 삽입
            video = distortion(video, corrupt_name, corrupt_severity, option = 'per_8_st')
            video = video.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
            
        elif corrupt == 'st_rotation': # st를 서로 상쇄되게끔 연속적으로 적용
            video = distortion(video, corrupt_name, corrupt_severity, option = 'st_rotation')
            video = video.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
            
        elif corrupt == "repeated_patterns_mix":
            video = repeated_patterns_mix(video) #TCHW, [0-1], float32
            
        elif corrupt == "repeated_patterns_one":
            video = repeated_patterns_one(video) #TCHW, [0-1], float32
            
        elif corrupt == "elastic_transform_incremental":
            video = elastic_transform_incremental(video)
            video = video.permute(0,3,1,2).contiguous().float() / 255.

        elif corrupt == "mix_incremental_temporal":
            video = mix_distortion(video, corrupt_severity = corrupt_severity, incremental =True, spatiotemporal=True)
            video = video.permute(0,3,1,2).contiguous().float() / 255.
        elif corrupt == "mix_incremental":
            video = mix_distortion(video, corrupt_severity = corrupt_severity, incremental = True, spatiotemporal=False)
            video = video.permute(0,3,1,2).contiguous().float() / 255.
        elif corrupt == "mix_fixed_temporal":
            video = mix_distortion(video, corrupt_severity = corrupt_severity, incremental = False, spatiotemporal=True)
            video = video.permute(0,3,1,2).contiguous().float() / 255.
        elif corrupt == "mix_fixed":
            video = mix_distortion(video, corrupt_severity = corrupt_severity, incremental = False, spatiotemporal=False)
            video = video.permute(0,3,1,2).contiguous().float() / 255.
        else:
            pass
            
    video = video.permute(1, 0, 2, 3).contiguous()  # TCHW -> CTHW

    return {'video': video}

# def repeated_pattern(video): #Input : THWC, [0-255]. uint8
#     #16Frame씩 Pattern이 반복된다고 가정
#     #1 -> 2 -> 3 ->4 | 1 -> 2 -> 3 -> 4

#     video = np.array(video, dtype=np.float32) / 255. #[0-1], float32

#     # 비디오의 복사본 생성
#     modified_video = copy.deepcopy(video)
    
#     # x = W, y = H, z = C
#     x,y,z = np.meshgrid(np.arange(video.shape[1]), np.arange(video.shape[0]), np.arange(video.shape[3]))

#     for frame_idx in range(65,128):
#         pattern = video[frame_idx-64]-video[frame_idx-65] #65frame-64frmae = 1frame - 0frame (64 frame 전 패턴 모방)
#         # dx = np.mean(pattern, axis = 2)* np.cos(np.pi/4).astype(np.float32) #2차원 [-1,1], float32
#         # dy = np.mean(pattern, axis = 2)* np.sin(np.pi/4).astype(np.float32)
#         dx, dy = dx[..., np.newaxis], dy[..., np.newaxis] #C축 만들어주기
#         indices = np.reshape(y + dy, (-1, 1)), \
#                     np.reshape(x + dx, (-1, 1)), \
#                     np.reshape(z, (-1, 1)) #1D 배열변환
#         #0~1 벗어날 수 있으니까 clip 해줌
#         modified_video[frame_idx] =  np.clip(map_coordinates(modified_video[frame_idx-1], indices, order=1, mode='reflect').reshape(video[0].shape),0,1)
    
#     modified_video = torch.from_numpy(modified_video * 255)

#     return modified_video

def repeated_patterns_mix(video): #Input : TCHW, [0-1], float32
    #16Frame씩 Pattern이 반복된다고 가정
    #1 -> 2 -> 3 -> 4 -> 4 -> -3 -> 2 -> -1
    
    modified_video = video.clone()
    
    # 패턴 정의
    pattern_1 = modified_video[:16]
    pattern_2 = modified_video[16:32]
    pattern_3 = modified_video[32:48]
    pattern_4 = modified_video[48:64]

    # 패턴을 modified_video에 적용
    modified_video[64:80] = pattern_4
    modified_video[80:96] = pattern_3.flip(0) #reverse the order of T dims
    modified_video[96:112] = pattern_2
    modified_video[112:] = pattern_1.flip(0)

    return modified_video

def repeated_patterns_one(video): #Input : TCHW, [0-1], float32
    #8 Frame씩 Pattern이 반복된다고 가정
    #1 ~ 8 -> 8 x 8
    modified_video = video.clone()
    # 패턴 정의
    pattern = modified_video[56:64]

    # 패턴을 modified_video에 적용
    for i in range(64,128,8):
        modified_video[i:i+8] = pattern 

    return modified_video

# def repeated_pattern_smooth(video, fade_length = 4): #Input : TCHW, [0-1], float32
#     #16Frame씩 Pattern이 반복된다고 가정
#     #1 -> 2 -> 3 -> 4 -> 4 -> -3 -> 2 -> -1
    
#     modified_video = video.clone()
    
#     # 패턴 정의
#     pattern_1 = modified_video[:16]
#     pattern_2 = modified_video[16:32]
#     pattern_3 = modified_video[32:48]
#     pattern_4 = modified_video[48:64]

#     # 패턴을 modified_video에 적용
#     modified_video[64:80] = pattern_4
#     modified_video[80:96] = pattern_3.flip(0) #reverse the order of T dims
#     modified_video[96:112] = pattern_2
#     modified_video[112:] = pattern_1.flip(0)

#     # 패턴 간 페이드 효과 추가 (코사인 보간 적용)
#     for i in range(fade_length):
#         alpha = (i + 1) / (fade_length + 1)
#         alpha = cosine_interpolation(alpha)
        
#         #pattern 4 -> pattern 4
#         modified_video[64 - fade_length + i] = (1 - alpha) * pattern_4[-fade_length + i] + alpha * pattern_4[i]
        
#         # pattern_4 -> pattern_3.flip(0)
#         modified_video[80 - fade_length + i] = (1 - alpha) * pattern_4[-fade_length + i] + alpha * pattern_3.flip(0)[i]
        
#         # pattern_3.flip(0) -> pattern_2
#         modified_video[96 - fade_length + i] = (1 - alpha) * pattern_3.flip(0)[-fade_length + i] + alpha * pattern_2[i]
        
#         # pattern_2 -> pattern_1.flip(0)
#         modified_video[112 - fade_length + i] = (1 - alpha) * pattern_2[-fade_length + i] + alpha * pattern_1.flip(0)[i]

#     return modified_video


def elastic_transform_incremental(video):
    corrupted_image_list = []
    for i in range(video.shape[0]): #128 frame, frame마다 severity 변화 or 16 frames 8구간
        # severity = np.linspace(0,1,128)[i]
        severity = np.linspace(0,1,8)[i//16] #0-15:0, 16-31: 1
        # corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = 'motion_blur_fraction', severity = severity, seed=42)
        corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = 'elastic_transform_fraction', severity = severity, seed=42)
        corrupted_image_list.append(torch.tensor(corrupted_image).unsqueeze(0)) #HWC -> THWC
    
    corrupted_video = torch.cat(corrupted_image_list, dim=0) #THWC
    
    return corrupted_video

def mix_distortion(video, corrupt_severity= None, incremental = False, spatiotemporal=False):
    corrupted_image_list = []
    
    for i in range(video.shape[0]): #128 frame, frame마다 severity 변화 or 16 frames 8구간
        # severity = np.linspace(0,1,128)[i]
        
        if incremental:
            severity = np.linspace(0,corrupt_severity,8)[i//16] #0-15:0, 16-31:1, 112-127:7
        else:
            severity = corrupt_severity
        
        if spatiotemporal == True:
            corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = 'motion_blur_fraction', severity = severity, seed=None)
            corrupted_image = corrupt(corrupted_image.astype(np.uint8), corruption_name = 'elastic_transform_fraction', severity = severity, seed=None)
        else:
            corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = 'motion_blur_fraction', severity = severity, seed=42)
            corrupted_image = corrupt(corrupted_image.astype(np.uint8), corruption_name = 'elastic_transform_fraction', severity = severity, seed=42)
            
        corrupted_image_list.append(torch.tensor(corrupted_image).unsqueeze(0)) #HWC -> THWC
    
    corrupted_video = torch.cat(corrupted_image_list, dim=0) #THWC
    
    return corrupted_video
  

def distortion(video, corruption_name, severity, seed = None, option = 'naive'):

    corrupted_image_list = []
    for i in range(video.shape[0]):
        #corrupt function returns CHW -> HWC,  #corrupt library input expects to HWC, numpy
        if option == "naive":
            corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=seed)
        elif option == "half_s_st":
            if i < (video.shape[0]//2): #s
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=42)
            else: #st
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=None)
        elif option == "last_16_st":
            if i >= (video.shape[0]-16): #st
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=None)
            else: #s
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=42)
        elif option == "per_8_st":
            if (i+1)%8 == 0: #st
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=None)
            else: #s
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=42)
        elif option == 'st_rotation': 
            assert corruption_name == 'motion_blur_custom'
            if i%4 == 0:
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=None, angle = 45)
            elif i%4 == 1:
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=None, angle = 135)
            elif i%4 == 2:
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=None, angle = -135)
            else:
                corrupted_image = corrupt(video[i,:,:,:].numpy().transpose(1, 2, 0), corruption_name = corruption_name, severity = severity, seed=None, angle = -45)
            
                 
        corrupted_image_list.append(torch.tensor(corrupted_image).unsqueeze(0)) #HWC -> THWC
    
    corrupted_video = torch.cat(corrupted_image_list, dim=0) #THWC

    return corrupted_video

def preprocess_image(image):
    # [0, 1] => [-1, 1]
    img = torch.from_numpy(image)
    return img

def cosine_interpolation(alpha):
    return (1 - math.cos(alpha * math.pi)) / 2

def smooth_transition(video, similar_videos: List[torch.Tensor], transition_length=8):
    C, T, H, W = video.shape
    total_frames = T
    num_videos = len(similar_videos) + 1
    segment_length = total_frames // num_videos
    remaining_frames = total_frames % num_videos

    current_video = video[:, :segment_length + remaining_frames, :, :]
    result = [current_video]

    for similar_video in similar_videos:
        next_segment = similar_video[:, :segment_length, :, :]
        transition_frames = []

        for i in range(transition_length):
            alpha = i / transition_length
            alpha = cosine_interpolation(alpha)  # Apply cosine interpolation
            
            current_frame = current_video[:, -1, :, :]
            next_frame = next_segment[:, 0, :, :]
            
            transition_frame = (1 - alpha) * current_frame + alpha * next_frame
            transition_frames.append(transition_frame.unsqueeze(1))

        transition_frames = torch.cat(transition_frames, dim=1)
        result.append(transition_frames)
        result.append(next_segment)

        current_video = next_segment

    result_video = torch.cat(result, dim=1)
    result_video = result_video[:, :128, :, :]

    return result_video

class VideoData(data.Dataset):
    """ Class to create dataloaders for video datasets 

    Args:
        data_path: Path to the folder with video frames or videos.
        image_folder: If True, the data is stored as images in folders.
        resolution: Resolution of the returned videos.
        sequence_length: Length of extracted video sequences.
        sample_every_n_frames: Sample every n frames from the video.
        batch_size: Batch size.
        num_workers: Number of workers for the dataloader.
        shuffle: If True, shuffle the data.
    """

    def __init__(self, data_path: str, image_folder: bool, resolution: int, sequence_length: int,
                 sample_every_n_frames: int, batch_size: int, num_workers: int, shuffle: bool = True, 
                 corrupt = None, corrupt_name = None, corrupt_severity = None):
        super().__init__()
        self.data_path = data_path
        self.image_folder = image_folder
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.sample_every_n_frames = sample_every_n_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.corrupt= corrupt
        self.corrupt_name = corrupt_name
        self.corrupt_severity = corrupt_severity

    def _dataset(self):
        '''
        Initializes and return the dataset.
        '''
        if self.image_folder:
            Dataset = FrameDataset
            dataset = Dataset(self.data_path, self.sequence_length,
                                resolution=self.resolution, sample_every_n_frames=self.sample_every_n_frames,
                                corrupt = self.corrupt, corrupt_name = self.corrupt_name, corrupt_severity = self.corrupt_severity)
        else:
            Dataset = VideoDataset
            dataset = Dataset(self.data_path, self.sequence_length,
                              resolution=self.resolution, sample_every_n_frames=self.sample_every_n_frames,
                              corrupt = self.corrupt, corrupt_name = self.corrupt_name, corrupt_severity = self.corrupt_severity)
        return dataset

    def _dataloader(self):
        '''
        Initializes and returns the dataloader.
        '''
        dataset = self._dataset()
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None and self.shuffle is True
        )
        return dataloader


class VideoDataset(data.Dataset):
    """ 
    Generic dataset for videos files stored in folders.
    Videos of the same class are expected to be stored in a single folder. Multiple folders can exist in the provided directory.
    The class depends on `torchvision.datasets.video_utils.VideoClips` to load the videos.
    Returns BCTHW videos in the range [0, 1].

    Args:
        data_folder: Path to the folder with corresponding videos stored.
        sequence_length: Length of extracted video sequences.
        resolution: Resolution of the returned videos.
        sample_every_n_frames: Sample every n frames from the video.
    """

    def __init__(self, data_folder: str, sequence_length: int = 16, resolution: int = 128, sample_every_n_frames: int = 1,
                 corrupt = None, corrupt_name = None, corrupt_severity = None):
        super().__init__()
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.sample_every_n_frames = sample_every_n_frames
        self.corrupt = corrupt
        self.corrupt_name = corrupt_name
        self.corrupt_severity = corrupt_severity
        self.folder = data_folder
        self.files = []
        self.labels = []
        self.label_dict = {}

        #video files, 하나의  큰 리스트로 합치기
        # files = sum([glob.glob(osp.join(folder, '**', f'*{ext}'), recursive=True)
        #              for ext in VID_EXTENSIONS], [])
        
        for i, (root, _, files) in enumerate(os.walk(self.folder)):
            if i>=1:
                label = os.path.basename(root)
                for file in files:
                    if file.endswith(tuple(VID_EXTENSIONS)):
                        file_path = osp.join(root, file)
                        self.files.append(file_path)
                        self.labels.append(label)
        
        self.label_dict = {file: label for file, label in zip(self.files, self.labels)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(self.folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            #sequence_length = clip_length in frames
            clips = VideoClips(self.files, sequence_length, num_workers=4)
            try:
                pickle.dump(clips.metadata, open(cache_file, 'wb'))
            except:
                print(f"Failed to save metadata to {cache_file}")
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(self.files, sequence_length,
                               _precomputed_metadata=metadata)
        
        self._clips = clips
        # instead of uniformly sampling from all possible clips, we sample uniformly from all possible videos
        self._clips.get_clip_location = self.get_random_clip_from_video

    def get_random_clip_from_video(self, idx: int) -> tuple:
        '''
        Sample a random clip starting index from the video.
        idx -> video idx, clip_id : random

        Args:
            idx: Index of the video.
        '''
        # Note that some videos may not contain enough frames, we skip those videos here.
        while self._clips.clips[idx].shape[0] <= 0:
            idx += 1
        n_clip = self._clips.clips[idx].shape[0]
        clip_id = random.randint(0, n_clip - 1) #random하게 clip 뽑히는 부분

        return idx, clip_id

    def __len__(self):
        return self._clips.num_videos()

    def __getitem__(self, idx):
        resolution = self.resolution
        while True:
            try:
                video, _, _, idx = self._clips.get_clip(idx)
            except Exception as e:
                print(idx, e)
                idx = (idx + 1) % self._clips.num_clips()
                continue
            break
        
        label = self.label_dict[self._clips.video_paths[idx]]
        
        video_label_dict = dict(**preprocess(video, resolution, sample_every_n_frames=self.sample_every_n_frames, corrupt = self.corrupt, corrupt_name = self.corrupt_name, corrupt_severity = self.corrupt_severity))
        video_label_dict['label'] = label
        # video_label_dict['similar_clips'] = similar_video_clips
        if self.corrupt == "smooth_transitions":
            #같은 label을 가지는 다른 clip.video_path 가져오기 -> 해당하는 index 번호를 가져와야함
            similar_video_paths = [path for path, l in self.label_dict.items() if l == label and path != self._clips.video_paths[idx]]
            similar_video_clips = []
            
            for path in similar_video_paths:
                # 해당 비디오 경로에 해당하는 VideoClips의 인덱스 찾기
                # sim_video 찾을때는 뭐 어떤 clip 위치를 고르든 상관 X
                similar_idx = self._clips.video_paths.index(path)
                try:
                    sim_video, _, _, _ = self._clips.get_clip(similar_idx)
                    sim_video = preprocess(sim_video, resolution, sample_every_n_frames=self.sample_every_n_frames)['video']
                    similar_video_clips.append(sim_video)
                except Exception as e:
                    print(f"Error retrieving similar clip from {path}: {e}")

            original_video = video_label_dict['video']
            video_label_dict['video'] = smooth_transition(original_video, similar_video_clips, transition_length = 4)
        # return dict(**preprocess(video, resolution, sample_every_n_frames=self.sample_every_n_frames, corrupt = self.corrupt, corrupt_name = self.corrupt_name, corrupt_severity = self.corrupt_severity))
        return video_label_dict #{'video': video, 'label' : label}


class FrameDataset(data.Dataset):
    """ 
    Generic dataset for videos stored as images. The loading will iterates over all the folders and subfolders
        in the provided directory. Each leaf folder is assumed to contain frames from a single video.

    Args:
        data_folder: path to the folder with video frames. The folder
            should contain folders with frames from each video.
        sequence_length: length of extracted video sequences
        resolution: resolution of the returned videos
        sample_every_n_frames: sample every n frames from the video
    """

    def __init__(self, data_folder, sequence_length, resolution=64, sample_every_n_frames=1,
                 corrupt = None, corrupt_name = None, corrupt_severity = None):
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.sample_every_n_frames = sample_every_n_frames
        self.data_all = self.load_video_frames(data_folder)
        self.video_num = len(self.data_all)
        self.corrupt = corrupt
        self.corrupt_name = corrupt_name
        self.corrupt_severity = corrupt_severity

    def __getitem__(self, index):
        batch_data = self.getTensor(index)
        return_list = {'video': batch_data}

        return return_list

    def load_video_frames(self, dataroot: str) -> list:
        '''
        Loads all the video frames under the dataroot and returns a list of all the video frames.

        Args:
            dataroot: The root directory containing the video frames.

        Returns:
            A list of all the video frames.

        '''
        data_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1]))
            except:
                print(meta[0], meta[2])
            if len(frames) < max(0, self.sequence_length * self.sample_every_n_frames):
                continue
            frames = [
                os.path.join(root, item) for item in frames
                if is_image_file(item)
            ]
            if len(frames) > max(0, self.sequence_length * self.sample_every_n_frames):
                data_all.append(frames)

        return data_all

    def getTensor(self, index: int) -> torch.Tensor:
        '''
        Returns a tensor of the video frames at the given index.

        Args:
            index: The index of the video frames to return.

        Returns:
            A BCTHW tensor in the range `[0, 1]` of the video frames at the given index.

        '''
        video = self.data_all[index]
        video_len = len(video)

        # load the entire video when sequence_length = -1, while the sample_every_n_frames has to be 1
        if self.sequence_length == -1:
            assert self.sample_every_n_frames == 1
            start_idx = 0
            end_idx = video_len
        else:
            n_frames_interval = self.sequence_length * self.sample_every_n_frames
            start_idx = random.randint(0, video_len - n_frames_interval) #랜덤으로 start_idx잡음
            end_idx = start_idx + n_frames_interval
        img = Image.open(video[0])
        h, w = img.height, img.width

        if h > w:
            half = (h - w) // 2
            cropsize = (0, half, w, half + w)  # left, upper, right, lower
        elif w > h:
            half = (w - h) // 2
            cropsize = (half, 0, half + h, h)

        images = []
        for i in range(start_idx, end_idx,
                       self.sample_every_n_frames):
            path = video[i]
            img = Image.open(path)

            if h != w:
                img = img.crop(cropsize)

            img = img.resize(
                (self.resolution, self.resolution),
                Image.Resampling.LANCZOS) #Image.ANTIALIAS is not supported after pillow ver 10.0.0
            
            if self.corrupt is None:
                img = np.asarray(img, dtype=np.float32)
                img /= 255.
                img_tensor = preprocess_image(img).unsqueeze(0) #numpy -> torch, Generate T axis
                images.append(img_tensor)
            
            else:
                img = np.asarray(img, dtype=np.uint8)
                img_tensor = preprocess_image(img).unsqueeze(0)
                images.append(img_tensor) 
                
        if self.corrupt is None:
            video_clip = torch.cat(images).permute(3, 0, 1, 2) # THWC -> CTHW
        
        else:
            video_clip = torch.cat(images).permute(0,3,1,2) #THWC -> TCHW
            
            if self.corrupt == 's':
                video_clip = distortion(video_clip, self.corrupt_name, self.corrupt_severity, seed=42, option = 'naive')
                video_clip = video_clip.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
                
            elif self.corrupt == 'st':
                video_clip  = distortion(video_clip , self.corrupt_name, self.corrupt_severity, seed=None, option= 'naive')
                video_clip  = video_clip.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
            
            elif self.corrupt == 'half_s_st':
                video_clip  = distortion(video_clip , self.corrupt_name, self.corrupt_severity, option = 'half_s_st')
                video_clip  = video_clip.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
            
            elif self.corrupt == 'last_16_st': #마지막 16 frame만 st
                video_clip  = distortion(video_clip , self.corrupt_name, self.corrupt_severity, option= 'last_16_st')
                video_clip  = video_clip.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
            
            elif self.corrupt == 'per_8_st': # 8 frame마다 st 삽입
                video_clip  = distortion(video_clip , self.corrupt_name, self.corrupt_severity, option = 'per_8_st')
                video_clip  = video_clip.permute(0,3,1,2).contiguous().float() / 255.  # THWC -> TCHW -> float()/255.
                

            video_clip = video_clip.permute(1, 0, 2, 3).contiguous() #TCHW -> #CTHW
            
        return video_clip  # A BCTHW tensor in the range `[0, 1]`

    def __len__(self):
        return self.video_num
