from email.policy import default
import os
import sys
import glob
import configargparse

p = configargparse.ArgumentParser()
p.add('--experiment_name', type=str, required=True)
p.add('--gpuid', default='0', help='GPU to use')
p.add('--batch_size', type=int, default=4)
p.add('--hn_in', type=int, required=True)
p.add('--dataset_size', type=int, default=-1)
p.add('--img_dim', type=int, default=100)
p.add('--val_img_dim', type=int, required=True)
p.add('--num_frames', type=int, default=25)
p.add('--dataset_root', type=str, help='path to video files', required=True)
p.add('--useclip', action='store_true')
p.add('--gauss_prior', action='store_true')
p.add('--real_ds_path', type=str, required=True)

p.add('--skipby', type=int, default='1')
p.add('--start', type=int, default=0)
p.add('--end', type=str, default='')

opt = p.parse_args()

ckpts = sorted(glob.glob(f'logs/{opt.experiment_name}/checkpoints/ckpt*.pth'), reverse=True)
ckpts = ckpts[::opt.skipby]
end = None if opt.end == '' else int(opt.end)
ckpts = ckpts[opt.start:end]

for ckpt in ckpts:
	print(f"$$$ ==== STARTING CHECKPOINT {ckpt} ==== $$$")
	
	command = f"CUDA_VISIBLE_DEVICES={opt.gpuid} "
	command += f"python experiment_scripts/train_video_fvd_validation.py "
	command += f"--experiment_name {opt.experiment_name} "
	command += f"--batch_size {opt.batch_size} --hn_in={opt.hn_in} "
	command += f"--dataset_size={opt.dataset_size} --img_dim {opt.img_dim} --val_img_dim {opt.val_img_dim} --num_frames {opt.num_frames} "
	command += f"--dataset_root={opt.dataset_root} "
	command += f"--checkpoint_path={ckpt} "
	if opt.useclip:
		command += f"--useclip "
	if opt.gauss_prior:
		command += f"--gauss_prior "
	command += f"--real_ds_path {opt.real_ds_path} "
	command += "--test"

	print(command)

	os.system(command)