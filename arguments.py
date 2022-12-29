import configargparse
import os
from functools import partial
import dataio, modules, loss_functions
import torch
from torch.utils.data import DataLoader

"""
Copying Files to a directory to know which version ran for a given run.
"""
def copy_py_files_to_exp_dir(exp_root):
    pyroot = f'pylogs/{exp_root}'
    os.makedirs(pyroot, exist_ok=True)
    os.system(f'cp *.py {pyroot}')

    experiment_scripts_pyroot = f'{pyroot}/experiment_scripts'

    os.makedirs(experiment_scripts_pyroot, exist_ok=True)
    os.system(f'cp experiment_scripts/*.py {experiment_scripts_pyroot}')

    print(f'Files copied to {pyroot}.')

def prechecks(opt):
    # copying files to this directory to know the exact version that was used for the run
    copy_py_files_to_exp_dir(opt.experiment_name)

    def process_img_dims(dim):
        return [int(x) for x in dim.split(",")] if "," in dim else [int(dim), int(dim)]

    opt.img_dim = process_img_dims(opt.img_dim)
    opt.val_img_dim = process_img_dims(opt.val_img_dim)

    # both CLIP and Gaussian are different forms of regularization. 
    # If using Gaussian regularization, CLIP should be turned off.
    if opt.gauss_prior:
        print('Using Gaussian Prior!')
        assert not opt.useclip

    return opt

def get_arguments():
    p = configargparse.ArgumentParser()

    p.add_argument('--logging_root', type=str, default='./logs', 
        help='root for logging')
    p.add_argument('--experiment_name', type=str, default='',
        help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

    # General training options
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4, 
        help='learning rate. default=1e-4')
    p.add_argument('--num_epochs', type=int, default=10000000,
        help='Number of epochs to train for.')

    # Options to save model and logs.
    p.add_argument('--epochs_til_ckpt', type=int, default=1,
        help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=500,
        help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--steps_til_fvd', type=int, default=14,
        help='Number of times the fvd will be computed, every xth summary step. Here, 14 x 500 = 7000 steps, where 500 is the steps for each summary.')

    # Model definition and inputs. 
    p.add_argument('--sample_frac', type=float, default=1.0,
        help='What fraction of video pixels to sample in each batch (default is all)')

    p.add_argument('--p_hidden_layers', type=int, default=3,
        help='Number of hidden layers in the primary network')
    p.add_argument('--p_hidden_dim', type=int, default=256,
        help='Dimension of the hidden layers in the primary network')

    p.add_argument('--hn_in', type=int, default=512,
        help='Input dimension to hidden layer')
    p.add_argument('--hn_hidden_layers', type=int, default=3,
        help='Number of hidden layers in the hyper network')
    p.add_argument('--hn_hidden_dim', type=int, default=256,
        help='Dimension of the hidden layers in the hyper network')

    p.add_argument('--std', type=float, default=0.01,
        help='standard deviation')

    p.add_argument('--img_dim', type=str, default="100,100",
        help='Resolution of the image')
    p.add_argument('--val_img_dim', type=str, default="128,128", 
        help='Resolution of the image')
    p.add_argument('--num_frames', type=int, default=16, 
        help='Length of the video')

    p.add_argument('--useclip', action='store_true', 
        help='use videoclip')
    p.add_argument('--gauss_prior', action='store_true', 
        help='Code to Prior')

    p.add_argument('--stage', type=int, default=1, 
        help='1: training, 2: fit latent')

    p.add_argument('--loss_fn', type=str, default="manhattan", 
        help='type of distance between the ground truth and the prediction. Please use "normalized_euclidean" for MNIST dataset')

    # Dataloader parameters
    p.add_argument('--dataset_size', type=int, default=-1,
        help='Number of videos to use for training')
    p.add_argument('--dataset_root', type=str, default='/scratch/bipasha31/How2Sign-Blobs/train/rhands_intervals_2s/*',
        help='dataset root')
    p.add_argument('--real_ds_path', type=str, default="/scratch/bipasha31/P8Bit37hlsQ-Full_128x128.zip", 
        help='fvd against the real dataset')
    p.add_argument('--num_val_samples', type=int, default=2096, 
        help='number of samples fvd computation')
    p.add_argument('--checkpoint_path', default=None, 
        help='Checkpoint to trained model')

    # Test
    p.add_argument('--random_data', action='store_true', help='Generate random data')
    p.add_argument('--test', action="store_true", help="generate videos and compute fvd")
    p.add_argument('--val', action="store_true", help="validation to reconstruct the learned videos")

    # Generic parameters
    p.add_argument('--print', action='store_true', help='print parameters to determine computational complexity')
    return prechecks(p.parse_args())

def parse_opts(opt):
    # the txt file containing the path for each video. An example can be found in "datasets"
    # the file should either have list of video paths  
    # or list of directories where the directories contain the video frames
    with open(opt.dataset_root) as r:
        video_paths = r.read().splitlines()
        if opt.dataset_size > -1:
            video_paths = video_paths[:opt.dataset_size]

    # Dataloaders
    train_dataset = dataio.Implicit3DWrapperMultipleVideos(
        sample_fraction=opt.sample_frac, 
        max_frame_len=opt.num_frames, 
        video_paths=video_paths,
        img_shape=opt.img_dim, 
        dataset_size=opt.dataset_size)

    test_dataset = dataio.Implicit3DWrapperMultipleVideos(
        sample_fraction=1.0, 
        max_frame_len=opt.num_frames, 
        video_paths=train_dataset.video_paths,
        img_shape=opt.val_img_dim, 
        dataset_size=opt.dataset_size)

    # Define the model.
    model = modules.VideoGen(in_features=3, out_features=train_dataset.channels, 
            num_instances=train_dataset.__len__(), mode="nerf", 
            type="relu", hn_hidden_features=opt.hn_hidden_dim, 
            hn_hidden_layers=opt.hn_hidden_layers, hn_in=opt.hn_in,
            hidden_features=opt.p_hidden_dim, num_hidden_layers=opt.p_hidden_layers,
            std=opt.std, useCLIP=opt.useclip)

    optim = torch.optim.Adam(lr=opt.lr, params=model.parameters())

    # path to store the checkpoints and logs.
    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    print(f'Training on {train_dataset.__len__()} items.')
    print(f'{opt.num_val_samples} samples will be generated for each validation step.')

    return train_dataset, test_dataset, model, optim, root_path

def get_dataloaders_from_dataset(opt, train_dataset, test_dataset):
    return [DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0), 
        DataLoader(test_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)]

def load_model_from_checkpoint(opt, model, optim):
    ckpt_dict = torch.load(opt.checkpoint_path, map_location="cpu")
    model_state_dict = ckpt_dict["model"]
    learned_latents = ckpt_dict["learned_latents"]
    if "optim" in ckpt_dict:
        optim.load_state_dict(ckpt_dict["optim"])

    try:
        model.videogen.load_state_dict(model_state_dict)
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
        
        # this is important for progressive training where we keep changing
        # the codebook size.
        latent_codes = load_missing(model, model_state_dict)

        # we then copy the latent codes learned in the previous stage 
        # to the current model.
        model.init_from_learned_latents(latent_codes)

    return learned_latents

def get_loss_fns(opt):
    # Define the loss
    lpips_train = loss_functions.LPIPSLoss(opt.img_dim).cuda()
    lpips_val = loss_functions.LPIPSLoss(opt.val_img_dim).cuda()

    loss_fns = {"train": partial(loss_functions.image_mse, lpips=lpips_train, loss=opt.loss_fn), 
        "val": partial(loss_functions.image_mse, lpips=lpips_val, loss=opt.loss_fn)}

    return loss_fns