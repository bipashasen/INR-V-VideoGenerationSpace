import os
import sys
import configargparse

stages = {
	1: {
		"ds_size": 10,
		"num_epochs": 710, # 7100
		"steps_til_summary": 500,
		"steps_til_fvd": 7, # 3500
		}, 
	2: {
		"ds_size": 100, 
		"num_epochs": 101, # 10100
		"steps_til_summary": 500,
		"checkpoint_dir": 10,
		"steps_til_fvd": 10, # 5000
		},
	3: {
		"ds_size": 1000,
		"num_epochs": 50, # 50000
		"steps_til_summary": 1000,
		"checkpoint_dir": 100,
		"steps_til_fvd": 7, # 7000
		},
	4: {
		"ds_size": 10000,
		"num_epochs": 10, # 100000
		"steps_til_summary": 1000,
		"checkpoint_dir": 1000,
		"steps_til_fvd": 8, #8000
		},
	5: {
		"ds_size": -1,
		"num_epochs": 100, 
		"steps_til_summary": 10000,
		"checkpoint_dir": 10000,
		"steps_til_fvd": 5, #10000
		},
	}

p = configargparse.ArgumentParser()
p.add('--ds', '--dataset', default='rainbow', help='Name of dataset')
p.add('--dstxt', default='rainbow-jelly-34k.txt', help='path to video files')
p.add('--gpuid', default='', help='GPU to use')
p.add('--useclip', action='store_true')
p.add('--stage', type=int, default=1)
p.add('--gauss_prior', action='store_true')
p.add('--real_ds_path', default='/scratch/bipasha31/P8Bit37hlsQ-Full_128x128.zip')
p.add('--num_frames', type=int, default=25)

opt = p.parse_args()

useclip = opt.useclip
if opt.gpuid != '':
	if opt.gpuid == "3":
		devices = "3"
	else:
		devices = f"{opt.gpuid},3"
else:
	devices = "1,3" if useclip else "2,3"

dataset = opt.ds
dataset_txt = opt.dstxt
h_in = 128
img_dim = 100
batch_size = 2

for stage in stages:
	if stage < opt.stage:
		continue

	print(f"$$$ ==== STARTING STAGE {stage} ==== $$$")
	params = stages[stage]

	ds_size = params['ds_size']
	num_epochs = params['num_epochs']
	steps_til_summary = params['steps_til_summary']
	steps_til_fvd = params['steps_til_fvd']

	suffix = ''
	if useclip:
		suffix = '-clip'
	elif opt.gauss_prior:
		suffix = '-var'

	experiment_name = f"ablations-{dataset}-{h_in}in-25f100d{suffix}"

	if "checkpoint_dir" in params:
		checkpoint_path = f"logs/{experiment_name}/{params['checkpoint_dir']}/checkpoints/model_final.pth"

	print(f"Parameters: {params}")

	command = f"CUDA_VISIBLE_DEVICES={devices} "
	command += f"python experiment_scripts/train_video_fvd_multi_gpu.py "
	command += f"--experiment_name {experiment_name}/{ds_size} "
	command += f"--model_type=nerf --batch_size {batch_size} --use_hn --hn_in={h_in} "
	command += f"--dataset_size={ds_size} --img_dim {img_dim} --num_frames {opt.num_frames} "
	command += f"--num_epochs={num_epochs} --steps_til_summary={steps_til_summary} --steps_til_fvd={steps_til_fvd} "
	command += f"--dataset_root=datasets/{dataset_txt} "
	if useclip:
		command += f"--useclip "
	if "checkpoint_dir" in params:
		command += f"--checkpoint_path={checkpoint_path} "
	if opt.gauss_prior:
		command += f"--gauss_prior "
	command += f"--real_ds_path {opt.real_ds_path}"

	print(command)

	os.system(command)
