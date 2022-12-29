'''Implements a generic training loop.
'''

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
import json
import constants

def save_videos(model_outputs, gts, i, write_sample_at, run_til, originals=None):

    base = f"{constants.save_minor_data_at}/samples/{write_sample_at}"
    os.makedirs(base, exist_ok=True)

    def get_video_as_numpy(x):
        try:
            return torch.clamp(x, min=-1.0, max=1.0).cpu().numpy()    
        except:
            return torch.clamp(x, min=-1.0, max=1.0).detach().cpu().numpy()

    def denorm(x):
        return (((x*0.5)+0.5) * 255).astype(np.uint8)

    model_outputs = get_video_as_numpy(model_outputs)[:run_til]
    if originals is not None:
        originals = get_video_as_numpy(originals)

    if gts is not None:
        gts = get_video_as_numpy(gts)[:run_til]

        for idx, (model_output, gt) in enumerate(zip(model_outputs, gts)):
            zfill_idx = str(idx).zfill(5)

            skvideo.io.vwrite(f"{base}/{i}_{zfill_idx}_pred.mp4", denorm(model_output))
            skvideo.io.vwrite(f"{base}/{i}_{zfill_idx}_gt.mp4", denorm(gt))

            if originals is not None:
                skvideo.io.vwrite(f"{base}/{i}_{zfill_idx}_original.mp4", denorm(originals[idx]))

    else:
        for idx, model_output in enumerate(model_outputs):
            zfill_idx = str(idx).zfill(5)

            skvideo.io.vwrite(f"{base}/{i}_{zfill_idx}_pred.mp4", denorm(model_output))

def validation_single_batch(model, loss_fn, model_input, gt, video_hw):
    gts = None
    val_losses = []
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

def validation_multiple_batches(model, loss_fn, model_input, gt, video_hwprod):
    val_losses, model_outputs, gts = [], [], []
    im_size = int(math.sqrt(video_hwprod))

    B = model_input['coords'].shape[0]
    batch_shape = (B, -1, im_size, im_size, 3)

    model_input_full = {key: value.cuda() for key, value in model_input.items()}
    model_input = {key: value.cuda() for key, value in model_input.items()}

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

def validation(model, write_sample_at, val_dataloader, loss_fn=None, epoch=0, step=0, use_pbar=True, run_til=100000000, std=0.01):
    val_losses = []

    model.eval()
    with torch.no_grad():
        try:
            video_hw = val_dataloader.dataset.dataset.shape[1:]
        except:
            video_hw = val_dataloader.dataset.shape[1:]

        video_hwprod = video_hw[0] * video_hw[1]

        hw_threshold_for_single_batch_render = 256**2

        if use_pbar: 
            val_dataloader = tqdm(val_dataloader)
        
        for val_step, (model_input, gt) in enumerate(val_dataloader):
            B = model_input['coords'].shape[0]
            shape = (B, -1, video_hw[0], video_hw[1], 3)
            originals = gt['original'].view(shape) if 'original' in gt else None
            
            if video_hwprod <= hw_threshold_for_single_batch_render:
                model_outputs, gts, c_val_losses = validation_single_batch(model, loss_fn, model_input, gt, video_hw)
            else:
                model_outputs, gts, c_val_losses = validation_multiple_batches(model, loss_fn, model_input, gt, video_hwprod)

            save_videos(model_outputs, gts, f"{epoch}_{step}_{val_step}", write_sample_at, run_til, originals=originals)

            val_losses.extend(c_val_losses)

            if val_step*B >= run_til:
                return val_losses

        return val_losses

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, write_sample_at, summary_fn, use_pbar=True,
          stage=1, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, std=0.01):

    if stage == 1:
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
    elif stage == 2:
        try:
            model_params = [(name, param) for name, param in model.named_parameters() if any([x in name for x in ['mu_latent_code', 'logvar_latent_code']])]
            optim = torch.optim.Adam(lr=lr, params=[p for _, p in model_params])
        except:
            model_params = [(name, param) for name, param in model.named_parameters() if any([x in name for x in ['latent_code']])]
            optim = torch.optim.Adam(lr=lr, params=[p for _, p in model_params])
    else:
        model_params = [(name, param) for name, param in model.named_parameters() if any([x in name for x in ['mu_latent_code', 'logvar_latent_code', 'hyper_net.nets.0', 'hyper_net.nets.1', 'hyper_net.nets.2']])]
        optim = torch.optim.Adam(lr=lr, params=[p for _, p in model_params])

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    learned_latents = {}

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses, kld_losses, disc_losses = [], [], []
        spk_losses, emotion_losses, intensity_losses = [], [], []

        training_start_time = time.time()
        time_elapsed_array = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt, std=std)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt, std=std)

                output_idx, output_mus, output_logvars, output_zs =\
                    model_output['idx'], model_output['mu'], model_output['logvar'], model_output['z']

                if model_output['mu'] is not None:
                    for idx, mu, logvar, z in zip(output_idx, output_mus, output_logvars, output_zs):
                        learned_latents[idx.item()] = [
                            mu.detach().cpu(),
                            logvar.detach().cpu(),
                            z.detach().cpu()
                        ]
                else:
                    for idx, z in zip(output_idx, output_zs):
                        learned_latents[idx.item()] = [
                            None,
                            None,
                            z.detach().cpu()
                        ]

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                if 'vqvae_latent' in model_output:
                    train_loss += model_output['vqvae_latent'].mean()

                train_losses.append(train_loss.item())
                # kld_losses.append(losses['kld_loss'].mean().item())
                # disc_losses.append(losses['disc_loss'].mean().item())
                # spk_losses.append(losses['spk_loss'].mean().item())
                # emotion_losses.append(losses['emotion_loss'].mean().item())
                # intensity_losses.append(losses['intensity_loss'].mean().item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))

                if not use_lbfgs:
                    # optim.zero_grad()
                    model.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    klmean, discmean = np.array(kld_losses).mean(), np.array(disc_losses).mean()
                    spkmean, emomean, intensitymean = np.array(spk_losses).mean(), np.array(emotion_losses).mean(), np.array(intensity_losses).mean()

                    tqdm.write("Epoch %d, Total loss %0.6f, KLD %0.06f, Disc %0.06f, Spk %0.06f, Emotion %0.06f, Intensity %0.06f, iteration time %0.6f"\
                        % (epoch, train_loss, klmean, discmean, spkmean, emomean, intensitymean, time.time() - start_time))
                    kld_losses, disc_losses = [], []
                    spk_losses, emotion_losses, intensity_losses = [], [], []

                    np.save(f'logs/{write_sample_at}/learned_latents', learned_latents)
                    
                    if val_dataloader is not None: # and total_steps % 200 == 0:
                        
                        val_losses = validation(model, write_sample_at, val_dataloader, loss_fn=loss_fn, epoch=epoch, step=step, use_pbar=use_pbar, run_til=5, std=std)

                        tqdm.write(f"Ran Validation Set... Total Steps {total_steps} Val loss {np.mean(val_losses)}")
                        writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        
                        model.train()

                total_steps += 1
                time_elapsed_array.append(training_start_time - time.time())
                np.savetxt(os.path.join(checkpoints_dir, 'timeelapsed_epoch_%04d.txt' % epoch),
                       np.array(time_elapsed_array))

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
