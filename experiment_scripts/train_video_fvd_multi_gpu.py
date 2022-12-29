import sys
import os
import torch

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import arguments, training_with_val_fvd

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Manager

from ptflops import get_model_complexity_info

"""
Multi GPU Helpers.
"""
def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

"""
Run
"""
def multigpu_train(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1499', world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)
    
    if not opt.print:
        assert opt.experiment_name != '' # You need to set the directory to store logs and models

    train_dataset, test_dataset, model, optim, root_path = arguments.parse_opts(opt)
        
    def create_dataloader_callback():
        train_dataloader, test_dataloader = arguments.get_dataloaders_from_dataset(
            opt, train_dataset, test_dataset)

        return train_dataloader, test_dataloader

    # Print model complexity and exit the code
    if opt.print:
        # Need to change module.py a bit to make this code work as this code expects 
        # your model to have a single tensor as input. Whereas in our implementation, 
        # the input is a dictionary.
        # NOTE: while computing the complexity, turn off any extra parameters like 
        # UseCLIP as these are not a part of the video generation model and are
        # only used to regularizing the video generation model.
        assert not opt.useclip

        macs, params = get_model_complexity_info(model, (25, 1048576, 3), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
        print("computational completxity: ", macs, "number of params: ", params)
        assert False

    # load model
    if opt.checkpoint_path:
        arguments.load_model_from_checkpoint(opt, model, optim)

    model.cuda()
    optimizer_to(optim, 'cuda')

    loss_fns = arguments.get_loss_fns(opt)

    training_with_val_fvd.multiscale_training(model=model, dataloader_callback=create_dataloader_callback,
        num_val_samples=opt.num_val_samples, optim=optim, real_ds_path=opt.real_ds_path,
        val_img_shape=opt.val_img_dim, max_frames=opt.num_frames, epochs=opt.num_epochs, 
        steps_til_summary=opt.steps_til_summary, gauss_prior=opt.gauss_prior, 
        epochs_til_checkpoint=opt.epochs_til_ckpt, std=opt.std, model_dir=root_path, loss_fns=loss_fns, 
        write_sample_at=opt.experiment_name, steps_til_fvd=(opt.steps_til_summary * opt.steps_til_fvd),
        rank=gpu, train_function=training_with_val_fvd.train, gpus=opt.gpus)


if __name__ == "__main__":
    manager = Manager()
    shared_dict = manager.dict()

    opt = arguments.get_arguments()
    opt.gpus = torch.cuda.device_count()

    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt, ))
    else:
        multigpu_train(0, opt)