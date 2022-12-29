import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import arguments, training_with_val_fvd

opt = arguments.get_arguments()

assert not opt.print, "if you want to print parameters and check the computational complexity then use `train_video_fvd_multi_gpu.py'"

train_dataset, test_dataset, model, optim, root_path = arguments.parse_opts(opt)
train_dataloader, test_dataloader = arguments.get_dataloaders_from_dataset(
    opt, train_dataset, test_dataset)

model.cuda()

if opt.checkpoint_path:
    learned_latents = arguments.load_model_from_checkpoint(opt, model, optim)

loss_fns = arguments.get_loss_fns(opt)

if opt.test or opt.val:
    assert opt.checkpoint_path 

    model.eval()
    checkpoints_dir = os.path.join(root_path, 'checkpoints')
    
    if "final" in opt.checkpoint_path:
        suffix = "_final"
    elif "current" in opt.checkpoint_path:
        suffix = "_current"
    else:
        epoch = opt.checkpoint_path.split('/')[-1].split('.')[0].split('_')[-1]
        suffix = "_" + epoch

    run_til = 5 if opt.test else 5000000

    training_with_val_fvd.summary(model, 0.0, checkpoints_dir, opt.experiment_name, opt.num_val_samples, opt.val_img_dim, learned_latents, test_dataloader, loss_fns["val"], opt.num_frames, opt.real_ds_path, suffix, 0, opt.test, opt.val, run_til)

else:
    training_with_val_fvd.train(model=model, train_dataloader=[train_dataloader, test_dataloader],
        optim=optim, epochs=opt.num_epochs, real_ds_path=opt.real_ds_path, steps_til_summary=opt.steps_til_summary, steps_til_fvd=(opt.steps_til_summary * opt.steps_til_fvd), epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path, loss_fns=loss_fns, write_sample_at=opt.experiment_name, gauss_prior=opt.gauss_prior, val_img_shape=opt.val_img_dim, max_frames=opt.num_frames, num_val_samples=opt.num_val_samples, std=opt.std)

