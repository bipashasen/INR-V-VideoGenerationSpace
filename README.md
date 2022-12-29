# INR-V: A Continuous Representation Space for Video-based Generative Tasks

[Bipasha Sen](https://bipashasen.github.io/)<sup>\*1</sup>,
[Aditya Agarwal](http://skymanaditya1.github.io/)<sup>\*1</sup>,
[Vinay Namboodiri](https://vinaypn.github.io/)<sup>2</sup>,
[C V Jawahar](https://faculty.iiit.ac.in/~jawahar/)<sup>1</sup><br>
<sup>1</sup>International Institute of Information Technology, Hyderabad, <sup>2</sup>University of Bath

<sup>\*</sup>denotes equal contribution

This is the official implementation of the paper "INR-V: A Continuous Representation Space for Video-based Generative Tasks" **accepted** at TMLR 2022.

For more results, information, and details visit our [**project page**](https://skymanaditya1.github.io/INRV/) and read our [**paper**](https://openreview.net/forum?id=aIoEkwc2oB&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)). Following is an example of **video interpolation**.

<img src="./outputs/interpolation-1.gif">
<img src="./outputs/interpolation-2.gif">

First and the last column are videos from the training dataset, the intermediate videos are generated using INR-V by interpolating in the learned latent space. We can see a smooth transition in content (identity, ornaments, spectacles) and motion (pose, mouth movements). 

## Getting started

1. Set up a conda environment with all dependencies using the following commands:

    ```
    conda env create -f environment.yml
    conda activate inrv
    ```

2. Download [**How2Sign-Faces**](http://cvit.iiit.ac.in/images/datasets/inrv/How2Sign-Blobs.tar) (downloadable link) dataset.

3. Untar the downloaded dataset and edit [```valid_how2sign_faces.txt```](datasets/valid_how2sign_faces.txt) in the [datasets](datasets) folder. Replace ```/scratch/bipasha31``` with your working directory in the file. Most of the experiments reported in the paper (Sec 5) are done using How2Sign-Faces. Rest of the datasets reported in the main paper will be released in sometime!

4. [```constants.py```](constants.py) file contains the working directory to store intermediate videos. Edit the two variables in the file to change the locations.

## Training INR-V

The following command trains the video-generation network. At set intervals, it will generate novel videos and compute the FVD. 

```
CUDA_VISIBLE_DEVICES=0,1,2 python experiment_scripts/train_video_fvd_multi_gpu.py
```
**Parameters**<br>
Full list of paramters can be found in [```arguments.py```](arguments.py)

```--experiment_name``` - name your experiment <br>
```--dataset_root``` - a list of datapoints where each datapoint is either a video or a folder with jpg that can be sorted using python's ```sorted``` function to recover the video.<br>
```--dataset_size``` - lets your select the first ```n``` many datapoints to use from the given list training datapoints. ```-1``` selects all the datapoints listed for training.<br>
```--img_dim``` and ```--val_img_dim``` - you can train the model and infer respectively the videos at the given resolutions.<br>
```--hn_in``` - for datasets with more than 2000 videos, put this as ```128```, otherwise ```64```.<br>
```--checkpoint_path``` - to resume training.<br>
```--useclip``` - remove if you do not want to use VideoCLIP (as reported in the paper Sec. 3.2) for training.<br>
```--steps_til_fvd``` - how often do you want to compute the FVD? Set this very very high in case you do not want to compute FVD at all.<br>
```--real_ds_path``` - the "zipped" dataset that contains folders, each folder has the frames of a video in a sorted order. Make sure the resolution of the dataset and the resolution of the videos generated using INR-V (```--val_img_dim```) is same. 

An example is given below: 

```
CUDA_VISIBLE_DEVICES=0,1,2 python experiment_scripts/train_video_fvd_multi_gpu.py --experiment_name=codeDebug --batch_size 1 --hn_in=128 --dataset_size=-1 --img_dim 100 --num_frames 25 --dataset_root=datasets/rainbow-jelly-34k-shuf2.txt --useclip --real_ds_path=/scratch/bipasha31/P8Bit37hlsQ-Full_128x128.zip
```

**Note**, in the above examples, we are using DEVICE=3 to compute the FVD. In case you are using the same GPU to train and test, we have notice the code runs out of memory at the time of FVD computation. Any fix is welcomed!

**Progressive training** (Sec. 3.3)

```
python progressive_training.py
```

## Testing trained checkpoints

If you want to **test all the checkpoints** in a trained experiment folder, you can use the following command. This command will generate novel samples for and print the FVD for each of the checkpoint.

```
python test_fvd_for_all.py
```

Make sure all the parameters like ```--hn_in``` and ```--useclip``` are set according to the training configuration, otherwise the model loading will throw errors. ```--experiment_name``` should be the name of the experiment folder, not the absolute/relation location. 

An example is given below:

```
python test_fvd_for_all.py --hn_in 128 --img_dim 100 --val_img_dim 128 --real_ds_path=/scratch/bipasha31/How2Sign-Blobs_128x128/train/faces_intervals_2s_common_folder.zip --experiment_name=h2gfaces --dataset_root=datasets/valid_how2sign_faces.txt  --useclip
```

If you want to **test for an individual checkpoint**, you can use the following command (follow the same guidelines as above). 

```
python experiment_scripts/train_video_fvd_validation.py
```

Here, use ```--checkpoint_path``` to locate the exact checkpoint. 

[**Pretrained Checkpoint** [FVD = 145.28]]() **(coming out soon!)** for How2Sign-Faces (score reported in the paper, **161.68**).

## Experiments (coming out soon!)

### Single-INR (Sec. 4.1)

### Video Interpolation (Sec. 5.1)

### Video Inversion (Sec. 5.3)

**Inversion**

**Inpainting**

**Frame-Prediction**

**Frame-Interpolation**

**Sparse-Inpainting**

**Superresolution**

## We would love your contributions to improve INR-V

INR-V is first of its kind which introduces a novel formulation of a video representation space that has the potential to be more representative than the existing video generation techniques. We introduce exciting video-generative tasks of video-inversion, video-inpainting, video superresolution, and many more, that have wide implications in the real world. However, inevitably it suffers from certain limitations and ***we would like to strongly encourage contributions and spur further research into some of the limitations listed below.*** 

1. **Training Time**: INR-V architecture is based on hypernetworks trained over INR functions of videos. Hypernetworks are notorious to train and therefore, it takes very long to train and get the best results. In the paper, we propose two techniques to overcome this limitation - CLIP regularization and progressive training. There is however a huge scope for improvement in terms of training time. 

2. **Diverse datasets**: To learn a useful representation space, INR-V relies on learning the underlying structure of a given dataset. INR-V showcases remarkable results on different datasets shown in the paper such as How2Sign-Faces, RainbowJelly and ToyDataset. Inevitably, given a dataset with limited structure and diverse datapoints (diverse classes, backgrounds, actions, etc.), INR-V fails to produce covincing results on such datasets (e.g., UCF101). Consequently, learning a strong representation space on such diverse datasets is an exciting line of research.  

3. **Low-resolution novel videos**: Although INR-V learns a strong representation space that enables several video-generative downstream tasks. However, the representation space learned by the network is sparse. Consequently, when videos are decoded from latent points that are sampled from sparse areas, the resulting video outputs are of low-resolution/quality. This makes it an area of improvement to learn a representation space that is dense, such that the novel videos sampled from this space are of high quality. 

## Thanks

### How2Sign-Faces

The dataset is processed from the [How2Sign dataset](https://how2sign.github.io/) and can be downloaded from the aforementioned link. We want to thank the authors of How2Sign for releasing this awesome dataset of expressive signers to encourage research in the direction of sign language understanding. 

### StyleGAN-V

We would like to thank the authors of [StyleGAN-V](https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf) ([https://github.com/universome/stylegan-v](https://github.com/universome/stylegan-v)) for releasing the code. We use their codebase to compute the FVD metric of the generated videos. 

## Citation
If you find our work useful in your research, please cite:
```
@article{
    sen2022inrv,
    title={{INR}-V: A Continuous Representation Space for Video-based Generative Tasks},
    author={Bipasha Sen and Aditya Agarwal and Vinay P Namboodiri and C.V. Jawahar},
    journal={Transactions on Machine Learning Research},
    year={2022},
    url={https://openreview.net/forum?id=aIoEkwc2oB},
    note={}
}
```

## Contact
If you have any questions, please feel free to email the authors.

Bipasha Sen: bipasha.sen@research.iiit.ac.in <br>
Aditya Agarwal: aditya.ag@research.iiit.ac.in <br>
Vinay Namboodiri: vpn22@bath.ac.uk <br>
C V Jawahar: jawahar@iiit.ac.in <br>
