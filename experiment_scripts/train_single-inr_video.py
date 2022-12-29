'''Reproduces Supplement Sec. 7'''

# Enable import from parent package
import sys
import os
import json
import torch
import numpy as np
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import skvideo.datasets

def copy_py_files_to_exp_dir(exp_root):
    pyroot = f'pylogs/{exp_root}'
    os.makedirs(pyroot, exist_ok=True)
    os.system(f'cp *.py {pyroot}')

    experiment_scripts_pyroot = f'{pyroot}/experiment_scripts'

    os.makedirs(experiment_scripts_pyroot, exist_ok=True)
    os.system(f'cp experiment_scripts/*.py {experiment_scripts_pyroot}')

    print(f'Files copied to {pyroot}.')

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=1510,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=500,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu)')
p.add_argument('--sample_frac', type=float, default=1.0,
               help='What fraction of video pixels to sample in each batch (default is all)')

p.add_argument('--img_dim', type=int, default=100, help='Resolution of the image')
p.add_argument('--num_frames', type=int, default=25, help='Length of the video')
p.add_argument('--dataset_size', type=int, default=-1,
               help='Number of videos to use for training')
p.add_argument('--dataset_root', type=str, help='dataset root',
               default='/scratch/bipasha31/How2Sign-Blobs/train/rhands_intervals_2s/*')

p.add_argument('--test', action='store_true', help='test only')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

main_experiment_name = opt.experiment_name

video_paths = None

if '.txt' in opt.dataset_root:
    with open(opt.dataset_root) as r:
        video_paths = r.read().splitlines()

for i, video_path in enumerate(video_paths):
    print("-"*60)
    print(f"#{i}/{len(video_paths)} Running for video: {video_path}")
    print("-"*60)

    opt.experiment_name = main_experiment_name + '/' + str(i)
    print(f"Experiment Name: {opt.experiment_name}")

    copy_py_files_to_exp_dir(opt.experiment_name)

    opt.write_sample = opt.experiment_name

    coord_dataset = dataio.Implicit3DWrapper50Videos(
        video_path, sample_fraction=opt.sample_frac, max_frame_len=opt.num_frames, img_shape=opt.img_dim)

    same_test_coord_dataset = dataio.Implicit3DWrapper50Videos(
        video_path, sample_fraction=opt.sample_frac, max_frame_len=opt.num_frames, img_shape=opt.img_dim)

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
    same_test_val_dataloader = DataLoader(same_test_coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    type, mode = 'relu', 'nerf'

    model = modules.SingleBVPNet(
        type=type, in_features=3, out_features=coord_dataset.channels).cuda()

    if opt.checkpoint_path:
        state_dict = torch.load(opt.checkpoint_path)

        model.videogen.load_state_dict(state_dict)

    # Define the loss
    loss_fn = partial(loss_functions.image_mse, None)
    summary_fn = partial(utils.write_video_summary, coord_dataset)

    if opt.test:
        print(f'Validating on {same_test_val_dataloader.dataset.__len__()} items.')

        training.validation(model=model, val_dataloader=same_test_val_dataloader, loss_fn=loss_fn, write_sample_at=opt.write_sample+'/val')

    else:
        print(f'Training on {dataloader.dataset.__len__()} items.')

        root_path = os.path.join(opt.logging_root, opt.experiment_name)

        training.train(model=model, train_dataloader=dataloader, val_dataloader=same_test_val_dataloader, epochs=opt.num_epochs, lr=opt.lr,
                       steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt, 
                       model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, write_sample_at=opt.write_sample)

