# Table of contents
1. [Movie example](#umap)
2. [Opus-DSD](#opusdsd)
    1. [80S ribosome](#80s)
    2. [Spliceosome](#splice)
3. [setup environment](#setup)
4. [prepare data](#preparation)
5. [training](#training)
6. [analyze result](#analysis)

# Movie of the PC1 of the latent space of 80 ribosome learned by opus-DSD <div id="umap">
The movie of 80 ribosome along PC1

https://user-images.githubusercontent.com/3967300/221396928-72303aad-66a1-4041-aabb-5d38a58cb7dd.mp4


# Opus-DSD <div id="opusdsd">
This repository contains the implementation of opus-deep structural disentanglement (DSD), which is developed by the research group of
Prof. Jianpeng Ma at Fudan University. The publication of this method is available at https://www.nature.com/articles/s41592-023-02031-6. There is a program on codeocean https://codeocean.com/capsule/9350896/tree/v1 for testing its inference. A newer version with better reconstruction quality is available upon request. An exemplar movie of the new version is shown below:


https://github.com/alncat/opusDSD/assets/3967300/810e85cc-445f-4e8c-bfde-e78fe87ec443


This program is built upon a set of great works:
- [cryoDRGN](https://github.com/zhonge/cryodrgn)
- [Neural Volumes](https://stephenlombardi.github.io/projects/neuralvolumes/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [Healpy](https://healpy.readthedocs.io/en/latest/)


This project seeks to unravel how a latent space, encoding 3D structural information, can be learned utilizing only 2D image supervisions which are aligned against a consensus reference model.

An informative latent space is pivotal as it simplifies data analysis by providing a structured and reduced-dimensional representation of the data.

Our approach strategically **leverages the inevitable pose assignment errors introduced during consensus refinement, while concurrently mitigating their impact on the quality of 3D reconstructions**. Although it might seem paradoxical to exploit errors while minimizing their effects, our method has proven effective in achieving this delicate balance.

The workflow of this method is demonstrated as follows:

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/workflow.png?raw=true "Opus-DSD Workflow")

Note that all input and output of this method are in real space! (Fourier space is good, but how about real space!)
The architecture of encoder is (Encoder class in cryodrgn/models.py):

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/encoder.png?raw=true "Opus-DSD encoder")


The architecture of decoder is (ConvTemplate class in cryodrgn/models.py. In this version, the default size of output volume is set to 192^3, I downsampled the intermediate activations to save some gpu memories. You can tune it as you wish, happy training!):

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/decoder.png?raw=true "Opus-DSD decoder")

## 80S ribosome <a name="80s"></a>
The weight file can be downloaded from https://www.icloud.com/iclouddrive/0fab8AGmWkNjCsxVpasi-XsGg#weights.
The other pkls for visualzing are deposited at https://drive.google.com/drive/folders/1D0kIP3kDhlhRx12jVUsZUObfHxsUn6NX?usp=share_link.
These files are from the epoch 16 and trained with output volume of size 192. ```z.16.pkl``` stores the latent encodings for all particles. ```ribo_pose_euler.pkl``` is the pose parameter file. Our program will read configurations from ```config.pkl```. Put them in the same folder, you can then follow the [analyze result](#analysis) section to visualize the latent space.

An exmaple UMAP of latent space for 80S ribosome:

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/umap-bold.png?raw=true "80S ribosome UMAP")

Comparison between some states:
![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/riborna.png?raw=true "80S ribosome rna swing")


A more colorful one, the particles are colored according to their projection classes, note the clusters often show certain dominant colors, this is due to the consensus refinement will account structural variations in images by distorting their pose paramters like **fitting a longer rod into a small gap by tilting the rod! (those are all what the consensus refinement can fit!)**

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/umapr.png?raw=true "80S ribosome color UMAP")

Data source: [EMPIAR-10002](https://www.ebi.ac.uk/empiar/EMPIAR-10002/). The particles are colored according to their pose parameters in this image.

## Spliceosome complex <a name="splice"></a>
UMAP and some selected classes for the spliceosome complex
<img width="1147" alt="image" src="https://github.com/alncat/opusDSD/assets/3967300/38bd6b99-f9f6-4cf9-b6d5-a22029d012d8">

The movie of the movement in spliceosome along PC1

https://user-images.githubusercontent.com/3967300/221396607-9e4d9882-86f8-4f54-8122-3ee7fe48c956.mp4

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

After the environment is sucessfully created, you can then activate it and execute our program within this environement.

```
conda activate dsd
```

The inference pipeline of our program can run on any GPU which supports cuda 10.2 and is fast to generate a 3D volume. However, the training of our program takes larger amount memory, we recommend using V100 GPUs at least.

# prepare data <a name="preparation"></a>

This program is developed based on cryoDRGN and adheres to a similar data preparation process. 

**Data Preparation Guidelines:**
1. **Cryo-EM Dataset:** Ensure that the cryo-EM dataset is stored in the MRCS stack file format. A good dataset for tutorial is the splicesome which is available at https://empiar.pdbj.org/entry/10180/ (It contains the consensus refinement result.)
   
2. **Consensus Refinement Result:** The program requires a consensus refinement result, which should not apply any symmetry and must be stored as a Relion STAR file. Other 3D reconstruction results such as 3D classification, as long as they determine the pose parameters of images, can also be supplied as input.

**Usage Example:**

Assuming the refinement result is stored as `consensus_data.star` and the format of the Relion STAR file is below version 3.0, 
You can then prepare the pose parameter file by executing the below command inside the opus-dsd folder:

```
python -m cryodrgn.commands.parse_pose_star /work/consensus_data.star -D 320 --Apix 1.699 -o sp-pose-euler.pkl
```
suppose the consensus_data.star is located at ```/work/``` directory, where

| argument | explanation|
| --- | --- |
| -D | the dimension of the particle image in your dataset|
| --Apix | is the angstrom per pixel of you dataset|
| -o | followed by the filename of pose parameter used by our program|
| --relion31 | include this argument if you are using star file from relion with version higher than 3.0|

Next, you can prepare the ctf parameter file by executing:

```
python -m cryodrgn.commands.parse_ctf_star /work/consensus_data.star -D 320 --Apix 1.699 -o sp-ctf.pkl -o-g sp-grp.pkl --ps 0
```
| argument | explanation|
| --- | --- |
| -o-g | used to specify the filename of ctf groups of your dataset|
| --ps |  used to specify the amount of phaseshift in the dataset|

For star file from relion with version hgiher than 3.0, you should add --relion31 and more arguments to the command line!

Checkout ```prepare.sh``` which combine both commands to save your typing.

Suppose you download the spliceosome dataset. You can prepare a particle stack named ```all.mrcs``` using

```relion_stack_create --i consensus_data.star --o all --one_by_one```

Finally, you should create a mask using the consensus model and RELION as in the ```postprocess```. The spliceosome dataset has a ```global_mask.mrc``` file. Suppose the filename of mask is ```mask.mrc```, move it to the program directory for simplicity.

After executing all these steps, you have all pkls required for running opus-DSD in the program directory ( You can specify any directories you like in the command arguments ).


# training <a name="training"></a>

When the pkls are available, you can then train the vae for structural disentanglement proposed in DSD using

```
python -m cryodrgn.commands.train_cv /work/all.mrcs --ctf ./sp-ctf.pkl --poses ./sp-pose-euler.pkl --lazy-single --pe-type vanilla --encode-mode grad --template-type conv -n 20 -b 12 --zdim 12 --lr 1.e-4 --num-gpus 4 --multigpu --beta-control 2. --beta cos -o /work/sp -r ./mask.mrc --downfrac 0.75 --valfrac 0.25 --lamb 2. --split sp-split.pkl --bfactor 4. --templateres 224
```

The second argument after train_cv specified the path of image stack.
The three arguments ```--pe-type vanilla --encode-mode grad --template-type conv``` ensure OPUS-DSD is enabled! The default values are setting to them in our program, you can ommit them incase of simplicity.

The function of each argument is explained as follows:
| argument |  explanation |
| --- | --- |
| --ctf   | ctf parameters of the image stack |
| --poses | pose parameters of the image stack |
| -n     | the number of training epoches, each training epoch loops through all images in the training set |
| -b     | the number of images for each batch on each gpu, depends on the size of available gpu memory|
| --zdim  | the dimension of latent encodings, increase the zdim will improve the fitting capacity of neural network, but may risk of overfitting |
| --lr    | the initial learning rate for adam optimizer, 1.e-4 should work fine. |
| --num-gpus | the number of gpus used for training, note that the total number of images in the total batch will be n*num-gpus |
| --multigpu |toggle on the data parallel version |
| --beta-control |the restraint strength of the beta-vae prior, the larger the argument, the stronger the restraint. The scale of beta-control should be propotional to the SNR of dataset. Suitable beta-control might help disentanglement by increasing the magnitude of latent encodings and the sparsity of latent encodings, for more details, check out [beta vae paper](https://openreview.net/forum?id=Sy2fzU9gl). In our implementation, we adjust the scale of beta-control automatically based on SNR estimation, possible ranges of this argument are [0.5-4.]. You can use larger beta-control for dataset with higher SNR|
| --beta |the schedule for restraint stengths, ```cos``` implements the [cyclic annealing schedule](https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/) and is the default option|
| -o | the directory name for storing results, such as model weights, latent encodings |
| -r | ***the solvent mask created from consensus model***, our program will focus on fitting the contents inside the mask (more specifically, the 2D projection of a 3D mask). Since the majority part of image dosen't contain electron density, using the original image size is wasteful, by specifying a mask, our program will automatically determine a suitable crop rate to keep only the region with densities. |
| --downfrac | the downsampling fraction of input image, the input to network will be downsampled to size D\*downfrac, where D is the original size. You can set it according to resolution of consensus model and the ***templateres*** you set. |
| --lamb | the restraint strength of structural disentanglement prior proposed in DSD, set it according to the SNR of your dataset, for dataset with high SNR such as ribosome, splicesome, you can set it to 1. or higher, for dataset with lower SNR, consider lowering it. Possible ranges are [0.1, 3.]. If you find **the UMAP of embeedings is exaggeratedly stretched into a ribbon**, then the lamb you used during training is too high! |
| --split | the filename for storing the train-validation split of image stack |
| --valfrac | the fraction of images in the validation set, default is 0.1 |
| --bfactor | will apply exp(-bfactor/4 * s^2 * 4*pi^2) decaying to the FT of reconstruction, s is the magnitude of frequency, increase it leads to sharper reconstruction, but takes longer to reveal the part of model with weak density since it actually dampens learning rate, possible ranges are [3, 8]. Consider using higher values for more dynamic structures. We will decay the bfactor slightly in every epoch. This is equivalent to learning rate warming up. |
| --templateres | the size of output volume of our convolutional network, it will be further resampled by spatial transformer before projecting to 2D images. The default value is 192. You may keep it around ```D*downfrac/0.75```, which is larger than the input size. This corresponds to downsampling from the output volume of our network. You can tweak it to other resolutions, larger resolutions can generate sharper density maps, ***choices are Nx16, where N is integer between 8 and 16*** |
| --plot | you can also specify this argument if you want to monitor how the reconstruction progress, our program will display the 2D reconstructions and experimental images after 8 times logging intervals. Namely, you switch to interative mode by including this. |
| --tmp-prefix | the prefix of intermediate reconstructions, default value is ```tmp```. OPUS-DSD will output temporary reconstructions to the root directory of this program when training, whose names are ```$tmp-prefix.mrc``` |


The plot mode will display the following images:

<img width="553" alt="image" src="https://github.com/alncat/opusDSD/assets/3967300/68c08944-d096-42a2-bc2f-ec18584f319e">


Each row shows a selected image and its reconstruction from a batch.
In the first row, the first image is the experimental image supplemented to encoder, the second image is a 2D reconstruction blurred by the corresponding CTF, the third image is the correpsonding experimental image after 2D masking.


You can use ```nohup``` to let the above command execute in background and use redirections like ```1>log 2>err``` to redirect ouput and error messages to the corresponding files.
Happy Training! Contact us when running into any troubles.

To restart execution from a checkpoint, you can use

```
python -m cryodrgn.commands.train_cv /work/all.mrcs --ctf ./sp-ctf.pkl --poses ./sp-pose-euler.pkl --lazy-single -n 20 --pe-type vanilla --encode-mode grad --template-type conv -b 12 --zdim 12 --lr 1.e-4  --num-gpus 4 --multigpu --beta-control 2. --beta cos -o /work/sp -r ./mask.mrc --downfrac 0.75 --lamb 2. --valfrac 0.25 --load /work/sp/weights.0.pkl --latents /work/sp/z.0.pkl --split sp-split.pkl --bfactor 4. --templateres 224
```
| argument |  explanation |
| --- | --- |
| --load | the weight checkpoint from the restarting epoch |
| --latents | the latent encodings from the restarting epoch |

boths are in the output directory

During training, opus-DSD will output temporary volumes called ```tmp*.mrc``` (or the prefix you specified), you can check out the intermediate results by viewing them in Chimera. Opus-DSD uses 3D volume as intermediate representation, so it requires larger amount of memory, v100 gpus will be sufficient for its training. Its training speed is slower, which requires 2 hours on 4 v100 gpus to finish one epoch on dataset with 20k images. Opus-DSD also reads all images into memory before training, so it may require some more host memories (but this behavior can be toggled off, i didn't add an argument yet)

# analyze result <a name="analysis"></a>
You can use the analysis scripts in opusDSD to visualizing the learned latent space! The analysis procedure is detailed as following. You can try out our program at https://codeocean.com/capsule/9350896/tree/v1, which is a slightly older version.

The first step is to sample the latent space using kmeans algorithm. Suppose the results are in ```/work/sp```,

```
sh analyze.sh /work/sp 16 4 16
```

- The first argument after ```analyze.sh``` is the output directory used in training, which stores ```weights.*.pkl, z.*.pkl, config.pkl```
- the second argument is the epoch number you would like to analyze,
- the third argument is the number of PCs you would like to sample for traversal
- the final argument is the number of clusters for kmeans clustering.

The analysis result will be stored in /work/sp/analyze.16, i.e., the output directory plus the epoch number you analyzed, using the above command. You can find the UMAP with the labeled kmeans centers in /work/sp/analyze.16/kmeans16/umap.png and the umap with particles colored by their projection parameter in /work/sp/analyze.16/umap.png .

After running the above command once, you can skip umap embedding step by appending the command in analyze.sh with ```--skip-umap```. Our analysis script will read the pickled umap embeddings directly.

You can generate the volume which corresponds to each cluster centroid or traverses the principal component using,
(you can check the content of scirpt first, there are two commands, one is used to evaluate volume at kmeans center, another one is for PC traversal, just choose one according to use case)

```
sh eval_vol.sh /work/sp 16 16 2.2 kmeans
```

- The first argument following eval_vol.sh is the output directory used in training, which stores ```weights.*.pkl, z.*.pkl, config.pkl``` and the clustering result
- the second argument is the epoch number you just analyzed
- the third argument is the number of kmeans clusters (or principal component) you used in analysis
- the fourth argument is the apix of the generated volumes, you can specify a target value
- the last argument specifies wether generating volumes for kmeans clusters or principal components, use ```kmeans``` for kmeans clustering, or ```pc``` for principal components

change to directory ```/work/sp/analyze.16/kmeans16``` to checkout the reference*.mrc, which are the reconstructions
correspond to the cluster centroids.

You can use

```
sh eval_vol.sh /work/sp 16 1 2.2 pc
```

to generate volumes along pc1. You can check volumes in ```/work/sp/analyze.16/pc1```. You can make a movie using chimerax's vseries feature.
**PCs are great for visulazing the main motions and compositional changes of marcomolecules, while KMeans reveals representative conformations in higher qualities.**

Finally, you can also retrieve the star files for images in each kmeans cluster using

```
sh parse_pose.sh /work/consensus_data.star 1.699 320 /work/sp 16 16
```

- The first argument after ```parse_pose.sh``` is the star file of all images
- The second argument is apix value of image
- The third argument is the dimension of image
- The fourth arugment is the output directory used in training
- The fifth argument is the epoch number you just analyzed
- The final argument is the number of kmeans clusters you used in analysis

change to directory ```/work/sp/analyze.16/kmeans16``` to checkout the starfile for images in each cluster.
