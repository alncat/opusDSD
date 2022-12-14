# Table of contents
1. [UMAP example](#umap)
2. [Opus-DSD](#opusdsd)
    1. [80S ribosome](#80s)
    2. [Hrd1/Hrd3 complex](#hrd)
3. [setup environment](#setup)
4. [prepare data](#preparation)
5. [training](#training)
6. [analyze result](#analysis)

We now released weights and latents for some examples [80S ribosome](#80s), [Hrd1/Hrd3](#hrd)! You can use them to visualize the latent spaces for those examples.

# UMAP of the latent space of 80 ribosome learned by opus-DSD <div id="umap">
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/umapr.png?raw=true "80S ribosome color UMAP")

Data source: [EMPIAR-10002](https://www.ebi.ac.uk/empiar/EMPIAR-10002/). The particles are colored according to their pose parameters in this image. The uneven distribution of pose parameters reflects the pose assignment errors introduced by performing consensus refinement on dynamical cryo-EM dataset. The pose assignment error refers to the difference between the pose parameter of the particle obtained by aligning with its ground-truth conformation and its pose parameter obtained by aligning with the consensus model. **If all particles can be aligned with its ground-truth conformation, the distribution of pose parameters for every conformation should be even and uniform (ideally). However, when all particles are aligned against a single model, the distribution of pose parameters for every conformation will be distorted accordingly!**

# Opus-DSD <div id="opusdsd">
This repository contains the implementation of opus-deep structural disentanglement (DSD), which is developed by the research group of
Prof. Jianpeng Ma at Fudan University. The manuscript of this method is available at https://www.biorxiv.org/content/10.1101/2022.11.22.517601v1.
This program is built upon a set of great works:
- [cryoDRGN](https://github.com/zhonge/cryodrgn)
- [Neural Volumes](https://stephenlombardi.github.io/projects/neuralvolumes/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [Healpy](https://healpy.readthedocs.io/en/latest/)

It aims to answer the question that how can we learn a latent space encoding 3D structural information using only 2D image supervisions! Such an informative latent space should make data analysis much easier! **Our method actually works by exploiting the unavoidable pose assignment errors brought by consensus refinement while reducing the impact of pose assignment errors on the quality of 3D reconstructions! (sounds contradictary but it works!)**

The workflow of this method is demonstrated as follows:

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/workflow.png?raw=true "Opus-DSD Workflow")

Note that all input and output of this method are in real space! (Fourier space is good, but how about real space!)
The architecture of encoder is (Encoder class in cryodrgn/models.py):

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/encoder.png?raw=true "Opus-DSD encoder")


The architecture of decoder is (ConvTemplate class in cryodrgn/models.py. In this version, the default size of output volume is set to 192^3, I downsampled the intermediate activations to save some gpu memories. You can tune it as you wish, happy hacking!):

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/decoder.png?raw=true "Opus-DSD decoder")

## 80S ribosome <a name="80s"></a>
The weight file can be downloaded from https://www.icloud.com/iclouddrive/0fab8AGmWkNjCsxVpasi-XsGg#weights.
The other pkls for visualzing are deposited at https://drive.google.com/drive/folders/1D0kIP3kDhlhRx12jVUsZUObfHxsUn6NX?usp=share_link.
These files are from the epoch 16 and trained with output volume of size 192. ```z.16.pkl``` stores the latent encodings for all particles. ```ribo_pose_euler.pkl``` is the pose parameter file. Our program will read configurations from ```config.pkl```. Put them in the same folder, you can then follow the [analyze result](#analysis) section to visualize the latent space.

An exmaple UMAP of latent space for 80S ribosome:

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/umap-bold.png?raw=true "80S ribosome UMAP")

Comparison between some states:
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/riborna.png?raw=true "80S ribosome rna swing")


40S subunit rotation
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/riborotation.png?raw=true "80S ribosome rotation")
Note that UMAP1 actually correlates with this movement!

A more colorful one, the particles are colored according to their projection classes, note the clusters often show certain dominant colors, this is due to the consensus refinement will account structural variations in images by distorting their pose paramters like **fitting a longer rod into a small gap by tilting the rod! (those are all what the consensus refinement can fit!)**

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/umapr.png?raw=true "80S ribosome color UMAP")

## Hrd1/Hrd3 complex <a name="hrd"></a>
The weight file can be downloaded from https://www.icloud.com/iclouddrive/040I4UJjIpsWaCD2by_Df7PRQ#weights.
The other pkls for running visualzing are deposited at https://drive.google.com/drive/folders/1WdBwl_oSiy7fYPa0_HHUMVGLL0JZxVRz?usp=share_link.
These files are from the epoch 15. ```z.15.pkl``` stores the latent encodings for all particles. ```hrd_pose_euler.pkl``` is the pose paramter file. Our program will read configurations from ```config.pkl```. Put them in the same folder, you can then follow the [analyze result](#analysis) section to visualize the latent space.

Another exmaple UMAP of latent space for Hrd1/Hrd3 complex [EMPIAR-10099](https://www.ebi.ac.uk/empiar/EMPIAR-10099/):
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/umapht.png?raw=true "Hrd1/Hrd3 UMAP")

Tile of some states:
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/hrd.png?raw=true "hrd 15")

We should not regard all missing densities at a certain contour level as compositional difference in those states unless the occupancy difference is of large scale or consistent across different contour levels, like state 4 can be safely regarded as a different class since the whole left upper corner is gone! The weak densities might also be caused by poor pose alignments due to the high flexibility in that region. Hence, those regions can only resolve to poor resolutions even using our disentanglement program. If you believe your dataset is very homogenous, then you should focus on verifying whether the learning result reflects certain consistent dynamics. The final note is that, deep learning is great, but always be critical about its result when applying it to highly noisy dataset such as cryo-EM images since it is not foolproof!


The corresponding UMAP shows the locations of states:
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/umaph15.png?raw=true "hrd umap 15")

Superposition of state 8 with state 3:
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/hrds.png?raw=true "hrd 15 superposition")

The superposition clearly demonstrated the relative displacements of the two Hrd3 subunits. We can then understand how this complex can only be determined to low resolution like 4.7 angstrom (consensus model) and the upper middle part shows blurred weak densities. This example demonstrates that DSD can resolve compositional changes and dynamics in a unified framework.

# set up environment <a name="setup"></a>

After cloning the repository, to run this program, you need to have an environment with pytorch and a machine with GPUs. The recommended configuration is a machine with 4 V100 GPUs.
You can create the conda environment for DSD using the spec-file in the folder and by executing

```
conda create --name dsd --file spec-file
```

or using the environment.yml file in the folder by executing

```
conda env create --name dsd -f environment.yml
```

After the environment is sucessfully created, you can then activate it and using it to execute our program.

```
conda activate dsd
```

The inference pipeline of our program can run on any GPU which supports cuda 10.2 and is fast to generate a 3D volume. However, the training of our program takes larger amount memory, we recommend using V100 GPUs at least.

# prepare data <a name="preparation"></a>

The program is implemented on the basis of cryoDRGN. The data preparation process is very similar to it. First of all, the cryo-EM dataset should be stored as a mrcs stack file. Secondly, it requires a consensus refinement result without applying any symmetry which is stored as a relion star file (or any other results such as 3D classification which determine the pose parameters of images). Suppose the refinement result is stored in ```run_data.star``` and the format of relion star file is not above relion3.0 .

You can then prepare the pose parameter file by executing the below command inside the opus-dsd folder:

```
python -m cryodrgn.commands.parse_pose_star /work/run_data.star -D 192 --Apix 1.35 -o hrd-pose-euler.pkl
```
suppose the run_data.star is located at ```/work/``` directory, where
| argument | explanation|
| --- | --- |
| -D | the dimension of your dataset|
| --Apix | is the angstrom per pixel of you dataset|
| -o | followed by the filename of pose parameter used by our program|
| --relion31 | always add this argument if you are using star file from relion with version higher than 3.0|

Next, you can prepare the ctf parameter file by executing:

```
python -m cryodrgn.commands.parse_ctf_star /work/run_data.star -D 192 --Apix 1.35 -o hrd-ctf.pkl -o-g hrd-grp.pkl --ps 0
```
| argument | explanation|
| --- | --- |
| -o-g | used to specify the filename of ctf groups of your dataset|
| --ps |  used to specify the amount of phaseshift in the dataset|

Checkout ```prepare.sh``` which combine both commands to save your typing.

Thirdly, you should put the path of image stack in a txt file, e.g.,

a file named 'hrd.txt' which contains

```
/work/hrd.mrcs
```
Opus-DSD will read image stacks from the specified path.

Finally, you should create a mask using the consensus model and RELION as in the traditional postprocess. Suppose the filename of mask is ```mask.mrc```, move it to the program directory for simplicity.

After executing all these steps, you have three pkls for running opus-DSD in the program directory ( You can specify other directories in the command arguments ).


# training <a name="training"></a>

With the pkls available, you can then train the vae for structural disentanglement proposed in DSD using

```
python -m cryodrgn.commands.train_cv hrd.txt --ctf ./hrd-ctf.pkl --poses ./hrd-pose-euler.pkl --lazy-single -n 20 --pe-type vanilla --group ./hrd-grp.pkl --encode-mode grad -b 18 --zdim 8 --lr 1.e-4 --template-type conv --num-gpus 4 --multigpu --beta-control 0.01 --beta cos -o /work/hrd -r ./mask.mrc --downfrac 0.5 --lamb 0.5 --log-interval 1800 --split hrd-split.pkl --bfactor 4.
```

The meaning of each argument is explained as follows:
| argument |  explanation |
| --- | --- |
| --ctf   | ctf parameters of the image stack |
| --poses | pose parameters of the image stack |
| -n     | the number of training epoches, each training epoch loops through all images in the training set |
| --group | ctf groups of the image stack |
| -b     | the number of images for each batch on each gpu, depends on the size of available gpu memory|
| --zdim  | the dimension of latent encodings, increase the zdim will improve the fitting capacity of neural network, but may risk of overfitting |
| --lr    | the initial learning rate for adam optimizer, 1.e-4 should work, but you may use larger lr for dataset with higher SNR |
| --num-gpus | the number of gpus used for training, note that the total number of images in the total batch will be n*num-gpus |
| --multigpu |toggle on the data parallel version |
| --beta-control |the restraint strength of the beta-vae prior, the larger the argument, the stronger the restraint. The scale of beta-control should be propotional to the SNR of dataset. Suitable beta-control might help disentanglement by increasing the magnitude of latent encodings, for more details, check out [beta vae paper](https://openreview.net/forum?id=Sy2fzU9gl). In our implementation, we adjust the scale beta-control automatically based on SNR estimation, possible ranges of this argument are [0.5-0.8]|
| --beta |the schedule for restraint stengths, ```cos``` implements the [cyclic annealing schedule](https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/) |
| -o | the directory name for storing results, such as model weights, latent encodings |
| -r | the solvent mask created from consensus model, our program will focus on fitting the contents inside the mask (more specifically, the 2D projection of a 3D mask). Since the majority part of image dosen't contain electron density, using the original image size is wasteful, by specifying a mask, our program will automatically determine a suitable crop rate to keep only the region with densities. |
| --downfrac | the downsampling fraction of image, the reconstruction loss will be computed using the downsampled image of size D\*downfrac. You can set it according to resolution of consensus model. We only support D\*downfrac >= 128 so far (I may fix this behavior later) |
| --lamb | the restraint strength of structural disentanglement prior proposed in DSD, set it according to the SNR of your dataset, for dataset with high SNR such as ribosome, splicesome, you can safely set it to 1., for dataset with lower SNR, consider lowering it if the training yields spurious result. Possible ranges are [0.1, 1.5]|
| --log-interval | the logging interval, the program will output some statistics after the specified steps, set is to multiples of num-gpus\*b |
| --split | the filename for storing the train-validation split of image stack |
| --valfrac | the fraction of images in the validation set, default is 0.1 |
| --bfactor | will apply exp(-bfactor/4 * s^2 * 4*pi^2) decaying to the FT of reconstruction, s is the magnitude of frequency, increase it leads to sharper reconstruction, but takes longer to reveal the part of model with weak density since it actually dampens learning rate, possible ranges are [2, 4] |
| --templateres | the size of output volume of our convolutional network, it will be further resampled by spatial transformer before projecting to 2D images. The default value is 192. You can tweak it to other resolutions, larger resolutions can generate smoother density maps when downsampled from the output volume |
| --plot | you can also specify this argument if you want to monitor how the reconstruction progress, our program will display the 2D reconstructions and experimental images after 8 times logging intervals |
| --tmp-prefix | the prefix of intermediate reconstructions, default is ```tmp``` |


The plot mode will display the following images:
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/hrd2d.png?raw=true "2D projections of Hrd1/Hrd3 complex")
Each row shows a selected image and its reconstruction from a batch.
In the first row, the first image is a 2D projection, the second image is a 2D reconstruction blurred by the corresponding CTF, the third image is the correpsonding experimental image after 2D masking.
In the second row, the first image is the experimental image supplemented to encoder, the second image is the 2D reconstruction, the third image is the correpsonding experimental image without masking.

You can use ```nohup``` to let the above command execute in background and use redirections like ```1>log 2>err``` to redirect ouput and error messages.
Happy Training! Contact us if you run into any troubles, since we may miss certain points when writing this tutorial.

To restart execution from a checkpoint, you can use

```
python -m cryodrgn.commands.train_cv hrd.txt --ctf ./hrd-ctf.pkl --poses ./hrd-pose-euler.pkl --lazy-single -n 20 --pe-type vanilla --group ./hrd-grp.pkl --encode-mode grad -b 18 --zdim 8 --lr 1.e-4 --template-type conv --num-gpus 4 --multigpu --beta-control 0.01 --beta cos -o /work/output -r ./mask.mrc --downfrac 0.5 --lamb 0.5 --log-interval 1800 --load /work/hrd/weights.0.pkl --latents /work/hrd/z.0.pkl --split hrd-split.pkl
```

- --load, the weight checkpoint from the restarting epoch
- --latents, the latent encodings from the restarting epoch

boths are in the output directory

During training, opus-DSD will output temporary volumes called ```tmp*.mrc``` (or the prefix you specified), you can check out the intermediate result by viewing them using Chimera. Opus-DSD uses 3D volume as intermediate representation, so it requires larger amount of memory, v100 gpus will be sufficient for its training. Its training speed is slower, which requires 2 hours on 4 v100 gpus to finish one epoch on dataset with 20k images. Opus-DSD also reads all images into memory before training, so it may require some more host memories (but this behavior can be toggled off, i didn't add an argument yet)

# analyze result <a name="analysis"></a>
You can use the analysis scripts in opusDSD to visualizing the learned latent space! The analysis procedure is detailed as following.

The first step is to sample the latent space using kmeans algorithm. Suppose the results are in ```./data/ribo```,

```
sh analyze.sh ./data/ribo 16 ./data/ribo/ribo_pose_euler.pkl 16
```

- The first argument after ```analyze.sh``` is the output directory used in training, which stores ```weights.*.pkl, z.*.pkl, config.pkl```
- the second argument is the epoch number you would like to analyze,
- the third argument is the path of the pose parameter file you created before, which is used to color particles
- the final argument is the number of clusters for kmeans clustering.

The analysis result will be stored in ./data/ribo/analyze.16, i.e., the output directory plus the epoch number you analyzed, using the above command. You can find the UMAP with the labeled kmeans centers in ./data/ribo/analyze.16/kmeans16/umap.png and the umap with particles colored by their projection parameter in ./data/ribo/analyze.16/umap.png .

After running the above command once, you can skip umap embedding step by appending the command in analyze.sh with ```--skip-umap```. Our analysis script will read the pickled umap directly.

You can generate the volume corresponds to each cluster centroid using

```
sh eval_vol.sh ./data/ribo/ 16 16 1.77
```

- The first argument after eval_vol.sh is the output directory used in training, which stores ```weights.*.pkl, z.*.pkl, config.pkl``` and the clustering result
- the second argument is the epoch number you just analyzed
- the third argument is the number of kmeans clusters you used in analysis
- the fourth argument is the apix of the generated volumes, you can specify a target value

change to directory ```./data/ribo/analyze.16/kmeans16``` to checkout the reference*.mrc, which are the reconstructions
correspond to the cluster centroids.

Finally, you can also retrieve the star files for images in each cluster using

```
sh parse_pose.sh run_data.star 1.77 240 ./data/ribo/ 16 16
```

- The first argument after ```parse_pose.sh``` is the star file of all images
- The second argument is apix value of image
- The third argument is the dimension of image
- The fourth arugment is the output directory used in training
- The fifth argument is the epoch number you just analyzed
- The final argument is the number of kmeans clusters you used in analysis

change to directory ```./data/ribo/analyze.16/kmeans16``` to checkout the starfile for images in each cluster.
