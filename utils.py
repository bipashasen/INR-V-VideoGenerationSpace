import os
import glob
import random
import re
import math

import skvideo
import numpy as np
import torch

num_interpolate_points = 50

def compute_fvd_from_txt(txt):
    with open(txt) as r:
        result = r.read()

    return float(re.search('{"fvd2048_16f": (.*)}, "metric"', result).group(1))

def kl_loss(mu, log_var, std=0.01):
    std = log_var.mul(0.5).exp_()

    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std) * std)
    q = torch.distributions.Normal(mu, std)

    return 0.05 * torch.distributions.kl_divergence(p, q).mean()

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

def save_videos(model_outputs, i, base, run_til=-1, gts=None):
    os.makedirs(base, exist_ok=True)

    def get_video_as_numpy(x):
        try:
            return torch.clamp(x, min=-1.0, max=1.0).cpu().numpy()    
        except:
            return torch.clamp(x, min=-1.0, max=1.0).detach().cpu().numpy()

    def denorm(x):
        return (((x*0.5)+0.5) * 255).astype(np.uint8)

    model_outputs = get_video_as_numpy(model_outputs)

    if gts is not None:
        gts = get_video_as_numpy(gts)[:run_til]

        for idx, (model_output, gt) in enumerate(zip(model_outputs, gts)):
            zfill_idx = str(idx).zfill(5)

            skvideo.io.vwrite(f"{base}/{i}_{zfill_idx}_pred.mp4", denorm(model_output))
            skvideo.io.vwrite(f"{base}/{i}_{zfill_idx}_gt.mp4", denorm(gt))

    else:
        for idx, model_output in enumerate(model_outputs):
            zfill_idx = str(idx).zfill(5)

            skvideo.io.vwrite(f"{base}/{i}_{zfill_idx}.mp4", denorm(model_output))

def generate_video_single_batch(model, model_input, video_hw, loss_fn=None, gt=None):
    gts, val_losses = None, []

    model_input = {key: value.cuda() for key, value in model_input.items()}
            
    B = model_input['coords'].shape[0]
    shape = (B, -1, video_hw[0], video_hw[1], 3)

    output = model(model_input)
    model_outputs = output['model_out'].view(shape)

    if loss_fn is not None:
        gt = {key: value.cuda() for key, value in gt.items()}
        gts = gt['img'].view(shape)

        val_loss = loss_fn(output, gt, std=0.01)
        val_losses.append(val_loss['img_loss'].item())

    return model_outputs, gts, val_losses

def generate_video_multiple_batch(model, model_input, video_hwprod, loss_fn=None, gt=None):
    val_losses, model_outputs, gts = [], [], []
    im_size = int(math.sqrt(video_hwprod))

    B = model_input['coords'].shape[0]
    batch_shape = (B, -1, im_size, im_size, 3)

    model_input_full = {key: value.cuda() for key, value in model_input.items()}

    if loss_fn is not None:
        gt_full = {key: value.cuda() for key, value in gt.items()}
        gt = {key: value.cuda() for key, value in gt.items()}

    assert model_input['coords'].shape[1] % video_hwprod == 0

    for i in range(0, model_input['coords'].shape[1], video_hwprod):
        model_input['coords'] = model_input_full['coords'][:, i:i+video_hwprod, :]

        shape = batch_shape[1:]

        model_output = model(model_input)
        model_outputs.append(model_output['model_out'].view(shape))

        if loss_fn is not None:
            gt['img'] = gt_full['img'][:, i:i+video_hwprod, :],

            val_loss = loss_fn(model_output, gt)
            val_losses.append(val_loss['img_loss'].item())
            gts.append(gt['img'].view(shape))

    gts = torch.vstack(gts).view(batch_shape) if loss_fn is not None else None

    return torch.vstack(model_outputs).view(batch_shape), gts, val_losses

def compute_fvd(d, real_ds_path, val_img_dim, suffix):
    d_jpg = d + '_jpg'

    os.makedirs(d_jpg, exist_ok=True)

    mp4s = glob.glob(d + '/*.mp4')

    for mp4 in mp4s:
        fname = mp4.split('/')[-1].split('.')[0]
        sub_d_jpg = os.path.join(d_jpg, fname)

        os.makedirs(sub_d_jpg, exist_ok=True)

        os.system(f'ffmpeg -loglevel panic -i {mp4} {sub_d_jpg}/%04d.png')

    os.system(f'zip -qq -r {d_jpg}.zip {d_jpg}')

    real = real_ds_path
    fake = f"{d_jpg}.zip"

    command = "CUDA_VISIBLE_DEVICES=3 python src/scripts/calc_metrics_for_dataset.py"
    command += f" --real_data_path={real} --fake_data_path={fake} --mirror 1 --gpus 1 --resolution {val_img_dim[1]}"
    command += " --metrics fvd2048_16f --verbose 0 --use_cache 0"

    print(command)

    os.chdir('stylegan-v')
    os.system(f"{command} > {d_jpg}.txt")
    os.chdir('..')

    return compute_fvd_from_txt(f'{d_jpg}.txt')