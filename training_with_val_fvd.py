import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import torch.distributed as dist

from copy import deepcopy as c

from dataio import get_mgrid

from utils import *
import constants 

save_at_generation = f"{constants.save_major_data_at}/samples"
save_at_reconstruction = f"{constants.save_minor_data_at}/samples"

"""
Multi GPU Helpers
"""
def average_gradients(model):
    """Averages gradients across workers"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def multiscale_training(train_function, dataloader_callback, **kwargs):
    model = kwargs.pop('model', None)
    org_model_dir = kwargs.pop('model_dir', None)

    dataloaders = dataloader_callback()
    print(f'model_dir: {org_model_dir}')

    train_function(dataloaders=dataloaders, model_dir=org_model_dir, model=model, **kwargs)

"""
Validation - Generate Novel Samples and Compute FVD
"""
def generation_validation(model, write_sample_at, num_val_samples, val_img_dims, learned_latents, max_frames, real_ds_path, epoch, step):
    B = range(4)

    codes = torch.vstack([learned_latents[x][2].unsqueeze(0) for x in sorted(learned_latents)])

    shape = (max_frames, val_img_dims[0], val_img_dims[1])
    mgrid = get_mgrid(shape, dim=3) # 43520000 x 3

    base_random_interpolated = f"{save_at_generation}/{write_sample_at}/random_interpolated/{epoch}_{step}"

    txt = base_random_interpolated + "_jpg.txt"
    # fvd must already be computed for this epoch
    if not os.path.exists(txt):
        with torch.no_grad():
            video_hwprod = val_img_dims[0] * val_img_dims[1]

            hw_threshold_for_single_batch_render = 256**2 # threshold for GPU with 12GB memory. 

            for val_step in tqdm(range(num_val_samples // len(B))):
                z_random_interpolated = torch.vstack([generate_interpolated_random_samples(codes).unsqueeze(0) for _ in B])

                mgrids = torch.vstack([mgrid.unsqueeze(0) for _ in B])
                model_input_random_interpolated = {'z': z_random_interpolated, 'coords': mgrids}
                
                if video_hwprod <= hw_threshold_for_single_batch_render:
                    model_outputs_random_interpolated = generate_video_single_batch(model, model_input_random_interpolated, val_img_dims)[0]
                else:
                    model_outputs_random_interpolated = generate_video_multiple_batch(model, model_input_random_interpolated, video_hwprod)[0]

                save_videos(model_outputs_random_interpolated, f"{val_step}", base_random_interpolated)

            fvd_random_interpolated = compute_fvd(base_random_interpolated, real_ds_path, val_img_dims, 'random_interpolated')
    else:
        fvd_random_interpolated = compute_fvd_from_txt(txt)
    
    return {"fvd_random_interpolated": fvd_random_interpolated}

"""
Validation - Reconstruct and Compute Loss on Reconstruction
"""
def reconstruction_validation(model, write_sample_at, val_dataloader, loss_fn, epoch, step, run_til):
    val_losses = []

    model.eval()
    with torch.no_grad():
        video_hw = val_dataloader.dataset.shape[1:]

        video_hwprod = video_hw[0] * video_hw[1]

        hw_threshold_for_single_batch_render = 256**2

        for val_step, (model_input, gt) in enumerate(val_dataloader):
            B = model_input['coords'].shape[0]
            
            if video_hwprod <= hw_threshold_for_single_batch_render:
                model_outputs, gts, c_val_losses = generate_video_single_batch(model, model_input, video_hw, loss_fn, gt)
            else:
                model_outputs, gts, c_val_losses = generate_video_multiple_batch(model, model_input, video_hwprod, loss_fn, gt)

            base = f"{save_at_reconstruction}/{write_sample_at}/{epoch}_{step}"
            save_videos(model_outputs, f"{val_step}", base, run_til=run_til, gts=gts)

            val_losses.extend(c_val_losses)

            # If you don't want to reconstruct all the examples in the dataset (usually there are many)
            # you can select "run_til" number of videos to reconstruct, compute loss on, and visualize
            if val_step*B >= run_til:
                return np.mean(val_losses)

        return np.mean(val_losses)

"""
Summary of the trained model (Validation Step)
"""
def summary(model, time_elapsed, checkpoints_dir, write_sample_at, num_val_samples, val_img_dim, learned_latents, val_dataloader, loss_fn, max_frames, real_ds_path, epoch, step, compute_fvd, compute_reconstruction, run_til=5):
    stats = f"Epochs {epoch}"
    stats += f" Total Steps {step}"

    if compute_fvd:
        fvds = generation_validation(model, write_sample_at, num_val_samples, val_img_dim, learned_latents=learned_latents, max_frames=max_frames, real_ds_path=real_ds_path, epoch=epoch, step=step)
        fvd_random_interpolated = fvds["fvd_random_interpolated"]

        stats += f" FVD_random_interpolated: {fvd_random_interpolated:.4f}"

    if compute_reconstruction:
        reconstruction_val_loss = reconstruction_validation(
            model, write_sample_at, val_dataloader, loss_fn=loss_fn, epoch=epoch, step=step, run_til=run_til)

        stats += f" Rec. Val Loss: {reconstruction_val_loss:.4f}"

    time_elapsed_days = time_elapsed // 86400
    time_elapsed_hours = time_elapsed // 3600 % 24
    time_elapsed_mins = time_elapsed // 60 % 60
    time_elapsed_seconds = time_elapsed % 60

    stats += f" time_elapsed: {time_elapsed}"
    stats += f" time_elapsed_str: {time_elapsed_days}d {time_elapsed_hours}h {time_elapsed_mins}m {time_elapsed_seconds:.3f}s."

    tqdm.write(stats)
    
    if compute_fvd:
        with open(os.path.join(checkpoints_dir, 'fvd_vs_time.txt'), 'a') as w:
            w.write(stats + "\n")

    return stats

"""
Train Loop
"""
def train(model, dataloaders, optim, epochs, real_ds_path, steps_til_summary, steps_til_fvd, 
        epochs_til_checkpoint, model_dir, loss_fns, write_sample_at, gpus=1, rank=0, 
        gauss_prior=False, val_img_shape=128, max_frames=25, num_val_samples=3000, std=0.01):
    
    train_dataloader, val_dataloader = dataloaders

    summaries_dir = os.path.join(model_dir, 'summaries')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    
    if rank == 0: 
        for x in [model_dir, summaries_dir, checkpoints_dir]:
            os.makedirs(x, exist_ok=True)
            
    writer = SummaryWriter(summaries_dir)

    learned_latents = {}

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses, time_elapsed_array = [], []

        training_start_time = time.time()

        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                suffix = '%04d' % epoch
                torch.save({
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "learned_latents": learned_latents
                    }, os.path.join(checkpoints_dir, f"ckpt_epoch_{suffix}.pth"))
                
                np.savetxt(os.path.join(checkpoints_dir, f'train_losses_epoch_{suffix}.txt'), np.array(train_losses))

            for _, (model_input, gt) in enumerate(train_dataloader):
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                model_output = model(model_input)
                losses = loss_fns["train"](model_output, gt, std=std)

                output_idx, output_zs = model_output['idx'], model_output['z']

                for idx, z in zip(output_idx, output_zs):
                    learned_latents[idx.item()] = [None, None, z.detach().cpu()]

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                if gauss_prior:
                    train_loss += kl_loss(torch.mean(model.latent_codes.weight), torch.std(model.latent_codes.weight)).mean()

                train_losses.append(train_loss.item())

                model.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)
                optim.step()

                pbar.update(1)
                pbar.set_description(f"Training Loss: {train_loss:.3f}")

                time_elapsed = round(time.time() - training_start_time, 3)

                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    torch.save({
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "learned_latents": learned_latents
                    }, os.path.join(checkpoints_dir, "ckpt_current.pth"))

                if not total_steps % steps_til_summary and rank == 0:
                    model.eval()                
                    
                    is_save_videos = not total_steps % steps_til_fvd
                    summary(model, time_elapsed, checkpoints_dir, 
                        write_sample_at, num_val_samples, val_img_shape, 
                        learned_latents, val_dataloader, loss_fns["val"], max_frames, 
                        real_ds_path, epoch, total_steps, is_save_videos, is_save_videos)
                    
                    model.train()

                total_steps += 1
                time_elapsed_array.append(time_elapsed)
                if rank ==0:
                    np.savetxt(os.path.join(checkpoints_dir, 'timeelapsed_epoch_%04d.txt' % epoch),
                       np.array(time_elapsed_array))

        torch.save({
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "learned_latents": learned_latents
                }, os.path.join(checkpoints_dir, "ckpt_final.pth"))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

