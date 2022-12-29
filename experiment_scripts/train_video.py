'''Reproduces Supplement Sec. 7'''

# Enable import from parent package
import sys
import os
import json
import torch
import numpy as np
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import arguments
import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import skvideo.datasets

opt = arguments.get_arguments()

assert not opt.print 
"""
if you want to print parameters and check the computational complexity
then use `train_video_fvd_multi_gpu.py'
"""

if opt.stage == 2 or opt.test:
    opt.use_pbar = False
    opt.lpips = None
    opt.loss_fn = 'euclidean'
else:
    if 'mnist' in opt.dataset_root:
        opt.loss_fn = 'normalized_euclidean'
    else:
        opt.loss_fn = 'manhattan'

    opt.use_pbar = True
    opt.lpips = loss_functions.LPIPSLoss(opt.batch_size, opt.num_frames, opt.img_dim).cuda()

video_paths = None
with open(opt.dataset_root) as r:
    video_paths = r.read().splitlines()

    if opt.dataset_size > -1:
        video_paths = video_paths[:opt.dataset_size]

    if opt.optimize_index > -1:
        video_paths = video_paths[opt.optimize_index:opt.optimize_index+1]

if opt.isMEAD and not opt.useclip:
    opt.conditional = True

if opt.isMEAD:
    coord_dataset = dataio.MEAD(num_frames=opt.num_frames)
    same_test_coord_dataset = dataio.MEAD(num_frames=opt.num_frames)

    meadlens = {
        'spks': len(coord_dataset.speakers),
        'emotions': len(coord_dataset.emotions),
        'intensities': len(coord_dataset.intensities)
    }
elif opt.stage == 2:
    meadlens = None

    coord_dataset = dataio.Implicit3DWrapperInversion(
        opt.dataset_root, opt.case, opt.dataset_size, mode='train', max_frame_len=opt.num_frames, video_paths=video_paths, img_shape=opt.img_dim)
    
    same_test_coord_dataset = dataio.Implicit3DWrapperInversion(
        opt.dataset_root, opt.case, opt.dataset_size, mode='val', max_frame_len=opt.num_frames, video_paths=video_paths, img_shape=opt.img_dim)
else:
    meadlens = None

    coord_dataset = dataio.Implicit3DWrapperMultipleVideos(
        opt.dataset_root, sample_fraction=opt.sample_frac, max_frame_len=opt.num_frames, video_paths=video_paths,
        img_shape=opt.img_dim, dataset_size=opt.dataset_size, sort_video_paths=opt.sort_video_paths, conditional=opt.conditional)
    
    same_test_coord_dataset = dataio.Implicit3DWrapperMultipleVideos(
        opt.dataset_root, sample_fraction=1.0, max_frame_len=opt.num_frames, 
        img_shape=opt.img_dim, dataset_size=opt.dataset_size, video_paths=coord_dataset.video_paths, conditional=opt.conditional)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
same_test_val_dataloader = DataLoader(same_test_coord_dataset, shuffle=False if opt.test else True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

chosen_dataset = coord_dataset

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    type, mode = opt.model_type, 'mlp'
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    type, mode = 'relu', opt.model_type
else:
    raise NotImplementedError

deblur = False

if deblur:
    model_class = modules.VQVAE2DeblurNet

else:
    model_class = modules.SingleBVPNet

model = model_class(type=type, in_features=3, 
        out_features=chosen_dataset.channels, num_instances=chosen_dataset.__len__(), 
        mode=mode, hn_hidden_features=opt.hn_hidden_dim, 
        hn_hidden_layers=opt.hn_hidden_layers, use_hn=opt.use_hn, hn_in=opt.hn_in,
        hidden_features=opt.p_hidden_dim, num_hidden_layers=opt.p_hidden_layers,
        variational_latent=opt.var_latent, only_hypernet=opt.only_hypernet,
        conditional=opt.conditional, out_classes=coord_dataset.__len__(),
        useCLIP=opt.useclip, isMEAD=opt.isMEAD, meadlens=meadlens)

# PRINT PARAMETERS and BREAK

from prettytable import PrettyTable

if opt.print:
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
        
    count_parameters(model)
    assert False

# PRINT PARAMETERS and BREAK

model.cuda()

if opt.test:
    assert opt.checkpoint_path is not None

learned_latents = None

if opt.checkpoint_path:
    state_dict = torch.load(opt.checkpoint_path)

    try:
        model.videogen.load_state_dict(state_dict)
        print(f'model loaded from {opt.checkpoint_path}')
    except:
        def load_missing(model, pretrained_dict):
            model_dict = model.state_dict()
            latent_codes = pretrained_dict['latent_codes.weight']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
            missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

            print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
            print('miss matched params:', missed_params)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            return latent_codes
            
        latent_codes = load_missing(model, state_dict)

    if not deblur and not opt.stage == 2:
        model.init_from_learned_latents(latent_codes)

# Define the loss
loss_fn = partial(loss_functions.image_mse, lpips=opt.lpips, loss=opt.loss_fn)
summary_fn = partial(utils.write_video_summary, chosen_dataset)

if opt.random_data:
    random_sample_dataset = dataio.RandomSampleDataset(opt.checkpoint_path, max_frame_len=opt.num_frames, 
        img_shape=opt.img_dim, hn_hidden_features=opt.hn_hidden_dim, std=opt.std, conditional=opt.conditional)
    dataloader = DataLoader(random_sample_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)

    print(f'Generating {dataloader.dataset.__len__()} random videos.')

    training.validation(model=model, val_dataloader=dataloader, write_sample_at=opt.experiment_name+'/random_samples')

elif opt.test:
    print(f'Validating on {same_test_val_dataloader.dataset.__len__()} items.')

    training.validation(model=model, val_dataloader=same_test_val_dataloader, loss_fn=loss_fn, write_sample_at=opt.experiment_name+'/val')

else:
    print(f'Training on {dataloader.dataset.__len__()} items.')

    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    training.train(model=model, train_dataloader=dataloader, val_dataloader=same_test_val_dataloader, epochs=opt.num_epochs, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt, std=opt.std, stage=opt.stage,
                   model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, write_sample_at=opt.experiment_name, use_pbar=opt.use_pbar)

