This repository contains the implementation of opus-deep structural disentanglement (DSD), which is developed by the research group of
Prof. Jianpeng Ma at Fudan University. The manuscript of this method is available at https://www.biorxiv.org/content/10.1101/2022.11.22.517601v1

# set up environment

After cloning the repository, to run this program, you need to have an environment with pytorch and a machine with GPUs.
You can create the conda environment using the spec-file.txt in the folder and by executing

```
conda create --name dsd --file spec-file.txt
```

After the environment is sucessfully created, you can then activate it and using it to execute our program.

```
conda activate dsd
```

# prepare data

The program is implemented on the basis of cryoDRGN https://github.com/zhonge/cryodrgn. The data preparation process is very similar to cryoDRGN. First of all, the cryo-EM dataset should be stored as a mrcs stack file. Secondly, it requires a consensus refinement result which is stored as a relion star file. Suppose the refinement result is stored in ```run_data.star``` and the format of relion star file is not above relion3.0 .

You can then prepare the pose parameter file by executing the below command inside the opus-dsd folder:

```
python -m cryodrgn.commands.parse_pose_star /work/run_data.star -D 192 --Apix 1.4 -o hrd-pose-euler.pkl
```
suppose the run_data.star is located at ```/work/``` directory, where
- D is the dimension of your dataset,
- Apix is the angstrom per pixel of you dataset,
- o is followed by the filename of pose parameter used by our program.

Next, you can prepare the ctf parameter file by executing:

```
python -m cryodrgn.commands.parse_ctf_star /work/run_data.star -D 192 --Apix 1.4 -o hrd-ctf.pkl -o-g mtr-grp.pkl --ps 0
```
- o-g is used to specify the filename of ctf groups of your dataset.
- ps is used to specify the amount of phaseshift in the dataset.

Thirdly, you should put the path of image stack in a txt file, e.g.,

a file named 'hrd.txt' which contains

```
/work/hrd.mrcs
```

Finally, you should create a mask using the consensus model and RELION as in the traditional postprocess. Suppose the filename of mask is ```mask.mrc```, move it to the program directory for simplicity.

After executing all these steps, you have three pkls for running opus-DSD in the program directory ( You can specify other directories in the command arguments ).


# training

With the pkls available, you can then train the vae for structural disentanglement proposed in DSD using

```
python -m cryodrgn.commands.train_cv hrd.txt --ctf ./hrd-ctf.pkl --poses ./hrd-pose-euler.pkl --lazy-single -n 20 --pe-type vanilla --group ./hrd-grp.pkl --encode-mode grad -b 18 --zdim 8 --lr 1.e-4 --template-type conv --num-gpus 4 --multigpu --beta-control 0.01 --beta cos -o /work/hrd -r ./mask.mrc --downfrac 0.5 --lamb 0.5 --log-interval 1800
```

The meaning of each argument is explained as follows:

- ctf, ctf parameters of the image stack
- poses, pose parameters of the image stack
- n, the number of training epoches, each training epoch uses the whole image stack
- group, ctf groups of the image stack
- b, the number of images for each batch on each gpu
- zdim, the dimension of latent encodings
- lr, the learning rate for adam optimizer
- num-gpus, the number of gpus used for training, note that the total number of images in the total batch will be n*num-gpus
- multigpu, toggle on the data parallel version
- beta-control, the restraint strength of the beta-vae prior, the larger the argument, the stronger the restraint
- beta, the schedule for restraint stengths, cos implements the cyclic annealing schedule as in https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
- o, the directory name for storing results, such as model weights, latent encodings
- r, the solvent mask created from consensus model, our program will focus on fitting the contents inside the mask
- downfrac, the downsampling fraction of image, the reconstruction loss will be computed using the downsampled image of size D*downfraco
- lamb, the restraint strength of structural disentanglement prior proposed in DSD, set it according to the SNR of your dataset, for dataset with high SNR such as ribosome, splicesome, you can safely set it to 1., for dataset with lower SNR, consider lowering it if the training yields suprious result.
- log-interval, the logging interval, the program will output some statistics after the specified steps

To restart execution from a checkpoint, you can use

```
python -m cryodrgn.commands.train_cv hrd.txt --ctf ./hrd-ctf.pkl --poses ./hrd-pose-euler.pkl --lazy-single -n 20 --pe-type vanilla --group ./hrd-grp.pkl --encode-mode grad -b 18 --zdim 8 --lr 1.e-4 --template-type conv --num-gpus 4 --multigpu --beta-control 0.01 --beta cos -o /work/output -r ./mask.mrc --downfrac 0.5 --lamb 0.5 --log-interval 1800 --load /work/hrd/weights.0.pkl --latents /work/hrd/z.0.pkl
```

- load, the weight checkpoint from the restarting epoch
- latents, the latent encodings from the restarting epoch

boths are in the output directory

# analyze result
The analysis scripts are in another program, cryoViz, availabel at https://www.github.com/alncat/cryoViz .
clone it and change to the directory contains cryoViz

```
sh analyze.sh /work/hrd 12 /work/opus-DSD/hrd-pose-euler.pkl 10
```

- The first argument after analyze.sh is the output directory used in training,
- the second argument is the number of epoch you would like to analyze,
- the third argument is the pose parameter you created before,
- the final argument is the number of clusters for kmeans clustering.

The analysis result will be stored in /work/hrd/analyze.12 for this example.

You can generate the volumes of each cluster centroids using

```
sh eval_vol.sh /work/hrd/ 12 10 1 8
```

- The first argument after eval_vol.sh is the output directory used in training,
- the second argument is the number of epoch you just analyzed
- the third argument is the number of kmeans clusters you used in analysis
- the fourth argument is apix (which actually will be ignored, it is a dummy variable!!)
- the final argument is the dimension of latent space

Finally, you can also retrieve the star files for images in each cluster using

```
sh parse_pose.sh run_data.star 1.4 192 /work/hrd/ 12 10
```

- The first argument after parse_pose.sh is the star file of all images
- The second argument is apix value of image
- The third argument is the dimension of image
- The fourth arugment is the output directory used in training
- The fifth argument is the number of epoch you just analyzed
- The final argument is the number of kmeans clusters you used in analysis
