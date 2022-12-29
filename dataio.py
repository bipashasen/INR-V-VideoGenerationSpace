import csv
import glob
import math
import os
import random
from tqdm import tqdm

import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
from skimage.transform import resize as skresize
import skimage.filters
import skvideo.io
import skimage.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def process_video_from_path(path, max_frames, shape):
    if os.path.isfile(path): # load video from video file
        zero2one = True
        video = skvideo.io.vread(path).astype(np.single)
    else: # load video from folder containing individual frames in a sorted order.
        zero2one = False
        video = np.vstack([np.expand_dims(skimage.io.imread(x), 0)\
            for x in sorted(glob.glob(path + '/*.png') + glob.glob(path + '/*.jpg'))[:max_frames+1]])

    if any([x in path for x in ['sky_timelapse', 'rainbow', 'P8Bit37hlsQ']]):
        video = video[:, :, 140:500, :] # cropping the videos in the dataset to square.

    # resize given shape
    video = np.vstack([np.expand_dims(skresize(frame, (shape[1:]),
           anti_aliasing=True), 0) for frame in video])

    if zero2one:
        video = video / 255.

    shape, channels = video.shape[:-1], video.shape[-1] # T x H X w
    
    T = shape[0]

    if T >= max_frames:
        video = video[:max_frames]
    else:
        assert False # each video should be at least "max_frames" long

    return video, shape, channels

class Video(Dataset):
    def __init__(self, path_to_video):
        super().__init__()
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            self.vid = skvideo.io.vread(path_to_video).astype(np.single) / 255.

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid

class RandomSampleDataset(torch.utils.data.Dataset):
    # run_type should be in ["random-interpolate", "validation", "interpolate"]
    def __init__(self, model_path, max_frame_len=16, run_type='random-interpolate', num_samples=3000, img_shape=100, std=0.01):
        self.max_frames = max_frame_len
        self.hn_hidden_features = 128

        self.std = std
        self.num_samples = num_samples

        self.shape = (self.max_frames, img_shape, img_shape)
        self.mgrid = get_mgrid(self.shape, dim=3) # 43520000 x 3
        self.run_type = run_type

        # std and mean
        dict = torch.load(model_path, map_location='cpu')
        try:
            # Interpolate between the points seen during reconstruction.
            bpath = model_path.rsplit('/', 2)[0]

            codes = np.load(os.path.join(bpath, 'learned_latents.npy'), allow_pickle=True).item()
            codes = torch.vstack([codes[x][2].unsqueeze(0) for x in sorted(codes)])
        except:
            print('Didn\'t find the codes in the npy file, using the model\'s weights')

            codes = dict['latent_codes.weight']

        self.codes = codes
        self.mean, self.std = torch.mean(codes, dim=0), torch.std(codes, dim=0)

        if run_type == 'interpolate':
            self.num_interpolate_points = 10

            id1, id2 = random.randint(0, codes.shape[0]-1), random.randint(0, codes.shape[0]-1)
            p1 = codes[id1]
            p2 = codes[id2]

            print('Interpolating between', id1, 'and', id2)

            self.points = self.interpolate_points(p1, p2)

        elif run_type == 'validation':
            self.codes = self.codes[:2048]

    def __len__(self):
        if self.run_type == 'interpolate':
            return self.num_interpolate_points
        elif self.run_type == 'validation':
            return len(self.codes)
        else:
            return self.num_samples

    def interpolate_points_lineraly(self, p1, p2):
        # interpolate ratios between the points
        ratios = np.linspace(0, 1, num=self.num_interpolate_points)
        # linear interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = (1.0 - ratio) * p1 + ratio * p2
            vectors.append(v)
        return vectors

    # spherical linear interpolation (slerp)
    def slerp(self, val, low, high):
        omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
        so = np.sin(omega)
        if so == 0:
            # L'Hopital's rule/LERP
            return (1.0-val) * low + val * high
        return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
     
    # uniform interpolation between two points in latent space
    def interpolate_points(self, p1, p2):
        # interpolate ratios between the points
        ratios = np.linspace(0, 1, num=self.num_interpolate_points)
        # linear interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = self.slerp(ratio, p1, p2)
            vectors.append(v)
        return vectors

    def generate_interpolated_random_samples(self):
        self.num_interpolate_points = 50

        id1, id2 = random.randint(0, self.codes.shape[0]-1),\
            random.randint(0, self.codes.shape[0]-1)
        p1 = self.codes[id1]
        p2 = self.codes[id2]

        points = self.interpolate_points(p1, p2)

        return points[random.randint(0, len(points)-1)]

    def generate_random_samples(self):
        dist = torch.distributions.Normal(self.mean, self.std)

        return dist.sample()

    def get_validation_run(self, idx):
        return self.codes[idx]

    def __getitem__(self, idx):
        if self.run_type == 'random-interpolate':
            z = self.generate_interpolated_random_samples()
        elif self.run_type == 'random':
            z = self.generate_random_samples()
        elif self.run_type == 'interpolate':
            z = self.points[idx]
        elif self.run_type == 'validation':
            z = self.get_validation_run(idx)
        else:
            print('Select a proper option out of random-interpolate/interpolate/random')
            assert False

        in_dict = {'z': z, 'coords': self.mgrid}

        return in_dict, []

class Implicit3DWrapperInversion(torch.utils.data.Dataset):
    def __init__(self, case, dataset_size, mode, max_frame_len=25, img_shape=100, video_paths=None):
        self.video_paths = video_paths[:dataset_size+1] if dataset_size > -1 else video_paths
        self.dataset_size = len(self.video_paths)

        self.mode = mode

        self.max_frames = max_frame_len
        self.img_shape = img_shape

        print(f"Loading dataloader for inversion case {case}")

        self.shape = (self.max_frames, img_shape, img_shape)
        self.mgrid = get_mgrid(self.shape, dim=3) # 43520000 x 3

        self.case = case

        self.channels = 3

        self.fixed_coord_set = self.get_fixed_coord_set(case)

        self.faulty_indices = []
        self.faulty_indices_path = 'faulty-inversion-videos.txt'
        if os.path.exists(self.faulty_indices_path):
            os.remove(self.faulty_indices_path)

    def __len__(self):
        return self.dataset_size

    def get_fixed_coord_set(self, case):
        path = f'{case}.npy'
        if os.path.exists(path):
            return np.load(path)

        if case == 'inversion':
            coord_idx = torch.arange(0, len(self.mgrid))

        elif case == 'inpainting':
            coord_idx = []
            for i in range(self.max_frames):
                start = i*self.img_shape*self.img_shape
                end = start + ((self.img_shape*self.img_shape) // 2)
                coord_idx.extend(range(start, end))
                print(start, end)

            coord_idx = torch.tensor(coord_idx)

        elif case == 'frame-prediction':
            coord_idx = torch.arange(0, 1*self.img_shape*self.img_shape)

        elif case == 'frame-interpolation':
            coord_idx = []
            for i in [0, self.max_frames-1]:
                start = i*self.img_shape*self.img_shape
                end = start + (self.img_shape*self.img_shape)
                coord_idx.extend(range(start, end))

            coord_idx = torch.tensor(coord_idx)

        elif case == 'sparse-inpainting':
            coord_idx = torch.randint(0, len(self.mgrid), (int(.25 * len(self.mgrid)),))

        elif case == 'superresolution':
            coord_idx = torch.arange(0, len(self.mgrid))

        np.save(path, coord_idx)

        return coord_idx

    def __getitem__(self, idx):
        assert os.path.exists(self.video_paths[idx])

        if idx in self.faulty_indices:
            return self.__getitem__(random.randint(0, self.__len__()-1))

        video, shape, channels = process_video_from_path(self.video_paths[idx], self.max_frames, self.shape)
        
        assert video.shape[0] == self.max_frames

        shape = (self.max_frames,) + shape[1:]

        shape = math.prod(shape)

        data = (torch.from_numpy(video) - 0.5) / 0.5 # basically normalized.
        data = data.view(-1, channels) # 43520000 x 3 (bike)

        original = data.clone()

        coord_idx = self.fixed_coord_set

        if coord_idx is not None:
            sample_data = np.ones_like(data) * data.max().item()
            sample_data[coord_idx] = data[coord_idx]

            data = sample_data

        if self.mode == 'val':
            coords = self.mgrid
        else:
            data = data[coord_idx, :]
            coords = self.mgrid[coord_idx, :]

        in_dict = {'idx': idx, 'coords': coords, 'img': None}
        gt_dict = {'idx': idx, 'img': data, 'original': original}

        return in_dict, gt_dict

class Implicit3DWrapperMultipleVideos(torch.utils.data.Dataset):
    def __init__(self, video_paths, sample_fraction=1., max_frame_len=16, img_shape=[100, 100], dataset_size=-1):
        
        self.video_paths = video_paths
        self.sample_fraction = sample_fraction # 0.0038
        self.dataset_size = dataset_size

        self.channels = 3

        self.max_frames = max_frame_len

        self.shape = (self.max_frames, img_shape[0], img_shape[1])
        self.mgrid = get_mgrid(self.shape, dim=3) # 43520000 x 3

    def __len__(self):
        return len(self.video_paths) if self.dataset_size == -1 else self.dataset_size

    def __getitem__(self, idx):
        assert os.path.exists(self.video_paths[idx])

        try:
            return self.return_item(idx)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def return_item(self, idx):
        video, shape, channels = process_video_from_path(self.video_paths[idx], self.max_frames, self.shape)

        assert video.shape[0] == self.max_frames

        shape = (self.max_frames,) + shape[1:]

        shape = math.prod(shape)

        N_samples = int(self.sample_fraction * shape)

        data = (torch.from_numpy(video) - 0.5) / 0.5 # basically normalized.
        data = data.view(-1, channels) # 43520000 x 3 (bike)

        clip_video = data.reshape(self.shape + (3,))
        clip_video = np.vstack([np.expand_dims(skresize(frame, (224, 224),
               anti_aliasing=True), 0) for frame in video])

        if self.sample_fraction < 1.:
            coord_idx = torch.randint(0, shape, (N_samples,))
            data = data[coord_idx, :]
            coords = self.mgrid[coord_idx, :]
        else:
            coords = self.mgrid
            data = data

        in_dict = {'idx': idx, 'coords': coords, 'clip_img': clip_video}
        gt_dict = {'idx': idx, 'img': data}

        return in_dict, gt_dict

class Implicit3DWrapper50Videos(torch.utils.data.Dataset):
    def __init__(self, video_path, sample_fraction=1., max_frame_len=25, img_shape=100):
        
        self.video_path = video_path

        self.sample_fraction = sample_fraction # 0.0038

        self.channels = 3

        self.max_frames = max_frame_len

        self.shape = (self.max_frames, img_shape, img_shape)
        self.mgrid = get_mgrid(self.shape, dim=3) # 43520000 x 3

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert os.path.exists(self.video_path)

        video, shape, channels = process_video_from_path(self.video_path, self.max_frames, self.shape)

        assert video.shape[0] == self.max_frames

        shape = math.prod((self.max_frames,) + shape[1:])

        N_samples = int(self.sample_fraction * shape)

        data = (torch.from_numpy(video) - 0.5) / 0.5 # basically normalized.
        data = data.view(-1, channels) # 43520000 x 3 (bike)

        if self.sample_fraction < 1.:
            coord_idx = torch.randint(0, shape, (N_samples,))
            data = data[coord_idx, :]
            coords = self.mgrid[coord_idx, :]
        else:
            coords = self.mgrid
            data = data

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict
