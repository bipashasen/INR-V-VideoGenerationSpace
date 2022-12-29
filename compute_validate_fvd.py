import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import math
import skvideo.io
import glob
import json
import re

import configargparse

import multiprocessing as mp

from shutil import copytree

import skimage.io

import asyncio

import random

from copy import deepcopy as c

from dataio import get_mgrid
from constants import save_major_data_at

save_at_samples = f"{save_major_data_at}/samples"

num_interpolate_points = 50

def save_videos(model_outputs, i, base):
    os.makedirs(base, exist_ok=True)

    def get_video_as_numpy(x):
        try:
            return torch.clamp(x, min=-1.0, max=1.0).cpu().numpy()    
        except:
            return torch.clamp(x, min=-1.0, max=1.0).detach().cpu().numpy()

    def denorm(x):
        return (((x*0.5)+0.5) * 255).astype(np.uint8)

    model_outputs = get_video_as_numpy(model_outputs)

    for idx, model_output in enumerate(model_outputs):
        zfill_idx = str(idx).zfill(5)

        skvideo.io.vwrite(f"{base}/{i}_{zfill_idx}.mp4", denorm(model_output))

        # jpg = f"{base}/{i}_{zfill_idx}_jpg"
        # os.makedirs(jpg, exist_ok=True)

        # for imid, image in enumerate(model_output):
        #     skimage.io.imsave(f"{jpg}/{str(imid).zfill(2)}.png", image)

def validation_single_batch(model, model_input, video_hw, device):
    model_input = {key: value.to(device) for key, value in model_input.items()}
            
    B = model_input['coords'].shape[0]
    shape = (B, -1, video_hw[0], video_hw[1], 3)

    output = model(model_input)
    model_outputs = output['model_out'].view(shape)

    return model_outputs

def validation_multiple_batches(model, model_input, video_hwprod, device):
    model_outputs = []
    im_size = int(math.sqrt(video_hwprod))

    B = model_input['coords'].shape[0]
    batch_shape = (B, -1, im_size, im_size, 3)

    model_input_full = {key: value.to(device) for key, value in model_input.items()}
    model_input = {key: value.to(device) for key, value in model_input.items()}

    assert model_input['coords'].shape[1] % video_hwprod == 0

    for i in range(0, model_input['coords'].shape[1], video_hwprod):
        model_input['coords'] = model_input_full['coords'][:, i:i+video_hwprod, :]

        shape = batch_shape[1:]

        model_output = model(model_input)
        model_outputs.append(model_output['model_out'].view(shape))

    return torch.vstack(model_outputs).view(batch_shape)

def compute_fvd(d, real_ds_path, suffix):

    root = f'{save_major_data_at}/FVDs/{suffix}'
    sub_d = os.path.join(root, d.split('/')[-1])
    os.makedirs(sub_d, exist_ok=True)

    os.system(f"cp {d}/*.mp4 {sub_d}")
    
    sub_d_jpg = sub_d + '_jpg'

    os.makedirs(sub_d_jpg, exist_ok=True)

    mp4s = glob.glob(sub_d + '/*.mp4')

    for mp4 in mp4s:
        fname = mp4.split('/')[-1].split('.')[0]
        sub_sub_d_jpg = os.path.join(sub_d_jpg, fname)

        os.makedirs(sub_sub_d_jpg, exist_ok=True)

        os.system(f'ffmpeg -loglevel panic -i {mp4} {sub_sub_d_jpg}/%04d.png')

    os.system(f'zip -qq -r {sub_d_jpg}.zip {sub_d_jpg}')

    real = real_ds_path
    fake = f"{sub_d_jpg}.zip"

    command = "CUDA_VISIBLE_DEVICES=1 python src/scripts/calc_metrics_for_dataset.py"
    command += f" --real_data_path={real} --fake_data_path={fake} --mirror 1 --gpus 1 --resolution 128"
    command += " --metrics fvd2048_16f --verbose 0 --use_cache 0"

    os.chdir('stylegan-v')
    os.system(f"{command} > {sub_d_jpg}.txt")
    os.chdir('..')

    with open(f'{sub_d_jpg}.txt') as r:
        result = r.read()

    return re.search('{"fvd2048_16f": (.*)}, "metric"', result).group(1)

# spherical linear interpolation (slerp)
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=num_interpolate_points)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    return vectors

def generate_interpolated_random_samples(codes):
    id1, id2 = random.randint(0, codes.shape[0]-1),\
        random.randint(0, codes.shape[0]-1)
    p1 = codes[id1]
    p2 = codes[id2]

    points = interpolate_points(p1, p2)

    return torch.tensor(points[random.randint(0, len(points)-1)])

def validation(model, write_sample_at, num_val_samples, model_dir, max_frames, img_shape, real_ds_path, epoch, step):
    B = range(4)

    device = 'cuda:1'

    # model = model.to(device)

    codes = np.load(os.path.join(model_dir, 'learned_latents.npy'), allow_pickle=True).item()
    codes = torch.vstack([codes[x][2].unsqueeze(0) for x in sorted(codes)])
    codes_mean, codes_std = torch.mean(codes, dim=0), torch.std(codes, dim=0)

    dist = torch.distributions.Normal(codes_mean, codes_std)

    shape = (max_frames, img_shape, img_shape)
    mgrid = get_mgrid(shape, dim=3) # 43520000 x 3

    with torch.no_grad():
        video_hw = [img_shape, img_shape]
        video_hwprod = img_shape**2

        hw_threshold_for_single_batch_render = 256**2

        for val_step in tqdm(range(num_val_samples // len(B))):
            z_random = torch.vstack([dist.sample().unsqueeze(0) for _ in B])
            z_random_interpolated = torch.vstack([generate_interpolated_random_samples(codes).unsqueeze(0) for _ in B])

            mgrids = torch.vstack([mgrid.unsqueeze(0) for _ in B])

            model_input_random = {'z': z_random, 'coords': mgrids}
            model_input_random_interpolated = {'z': z_random_interpolated, 'coords': mgrids}
            
            if video_hwprod <= hw_threshold_for_single_batch_render:
                model_outputs_random = validation_single_batch(model, model_input_random, video_hw, device)
                model_outputs_random_interpolated = validation_single_batch(model, model_input_random_interpolated, video_hw, device)
            else:
                model_outputs_random = validation_multiple_batches(model, model_input_random, video_hwprod, device)
                model_outputs_random_interpolated = validation_multiple_batches(model, model_input_random_interpolated, video_hwprod, device)

            base_random = f"{save_at}/{write_sample_at}/random/{epoch}_{step}"
            base_random_interpolated = f"{save_at}/{write_sample_at}/random_interpolated/{epoch}_{step}"

            save_videos(model_outputs_random, f"{val_step}", base_random)
            save_videos(model_outputs_random_interpolated, f"{val_step}", base_random_interpolated)

        fvd_random = compute_fvd(base_random, real_ds_path, 'random')
        fvd_random_interpolated = compute_fvd(base_random_interpolated, real_ds_path, 'random_interpolated')

        return {"fvd_random": fvd_random, "fvd_random_interpolated": fvd_random_interpolated}

p = configargparse.ArgumentParser()

p.add_argument('--model_path')
p.add_argument('--time_elapsed')
p.add_argument('--checkpoints_dir')
p.add_argument('--write_sample_at', type=int)
p.add_argument('--num_val_samples', type=int)
p.add_argument('--model_dir')
p.add_argument('--max_frames', type=int)
p.add_argument('--img_shape', type=int)
p.add_argument('--real_ds_path')
p.add_argument('--epoch', type=int)
p.add_argument('--step', type=int)

model = modules.SingleBVPNet(type=type, in_features=3, 
        out_features=3, num_instances=opt.num_instances, 
        mode=opt.mode, hn_hidden_features=opt.hn_hidden_dim, 
        hn_hidden_layers=opt.hn_hidden_layers, use_hn=opt.use_hn, hn_in=opt.hn_in,
        hidden_features=opt.p_hidden_dim, num_hidden_layers=opt.p_hidden_layers,
        variational_latent=opt.var_latent, only_hypernet=opt.only_hypernet,
        conditional=opt.conditional, out_classes=coord_dataset.__len__(),
        useCLIP=opt.useclip, isMEAD=opt.isMEAD, meadlens=False)

time_elapsed = opt.time_elapsed
checkpoints_dir = opt.checkpoint_path
write_sample_at = opt.write_sample_at
num_val_samples = opt.num_val_samples
model_dir = opt.model_dir
max_frames = opt.max_frames
img_shape = opt.img_shape
real_ds_path = opt.real_ds_path
epoch = opt.epoch
step = opt.step

fvds = validation(model, write_sample_at, num_val_samples, model_dir=model_dir, max_frames=max_frames, img_shape=img_shape, real_ds_path=real_ds_path, epoch=epoch, step=step)

fvd_random = fvds["fvd_random"]
fvd_random_interpolated = fvds["fvd_random_interpolated"]

time_elapsed_days = time_elapsed // 86400
time_elapsed_hours = time_elapsed // 3600 % 24
time_elapsed_mins = time_elapsed // 60 % 60
time_elapsed_seconds = time_elapsed % 60

stats = f"Epochs {epoch}"
stats += f" Total Steps {step}"
stats += f" FVD_random: {fvd_random}"
stats += f" FVD_random_interpolated: {fvd_random_interpolated}"
stats += f" time_elapsed: {time_elapsed}"
stats += f" time_elapsed_str: {time_elapsed_days}d {time_elapsed_hours}h {time_elapsed_mins}m {time_elapsed_seconds}s."

tqdm.write(f"Ran Validation Set... {stats}")
with open(os.path.join(checkpoints_dir, 'fvd_vs_time.txt'), 'a') as w:
    w.write(stats)
