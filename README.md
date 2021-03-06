> This repository contains complementary material for the corresponding submission.

> **[Version of the document submission](submission/chavez-garcia_et_al_RAL.pdf)** with **high-quality figures**.

> **[Document with difference report](submission/diff_man.pdf)**

## Abstract

Mobile ground robots operating on unstructured terrain must predict which areas of the environment they are able to pass in order to plan feasible paths.
We address traversability estimation as a heightmap classification problem: we build a convolutional neural network that, given an image representing the heightmap of a terrain patch, predicts whether the robot will be able to traverse such patch from left to right.
The classifier is trained for a specific robot model (wheeled, tracked, legged, snake-like) using simulation data on procedurally generated training terrains; the trained classifier can be applied to unseen large heightmaps to yield oriented traversability maps, and then plan traversable paths. We extensively evaluate the approach in simulation on six real-world elevation datasets, and run a real-robot validation in one indoor and one outdoor environment.


## Repository content:

<!--**This repository provides:**-->
1. [Media](#media) material of the experiments on the real robot and real elevation maps,
2. [Data files](#data) (heightmaps and csv) to generate the training and evaluation datasets,
3. and the [source code](#code) to each module of our traversability estimation framework:
   * simulation and data generation,
   * heightmap generation,
   * dataset generation,
   * definition, training and evaluation of the CNN classifier,
   * generation of traversability representations such as oriented traversability maps and minimal traversability maps,
   * and visualization of the results as 3D renderings
   > verify the [software requirements](#software-requirements) to test the code

## Media



This is a selection of available media of our traversability estimation framework:

**Reachability overlay and paths for selected points on the Slope map**

{% include vimeoPlayer.html id=247479519 %}

**Minimal traversability overlay on the Sullens map**

{% include vimeoPlayer.html id=247478850 %}

**Demonstration of the experiments on the real Pioneer 3AT robot**

{% include vimeoPlayer.html id=224311562 %}

<!--[![video demonstration of the experiments on the real robot](https://i.vimeocdn.com/video/643340195_640.webp)](https://vimeo.com/224311562 "Pioneer 3AT in real scenario")-->

**Minimal traversability overlay on the Quarry map**

{% include vimeoPlayer.html id=224311774 %}

<!--[![animation of the minimal traversability map for the quarry dataset](https://i.vimeocdn.com/video/643336616_640.webp)](https://vimeo.com/224311774 "Minimal traversability map for the quarry dataset")-->


**Oriented traversability overlays for 32 orientations on the Quarry map**

{% include vimeoPlayer.html id=224311892 %}

<!--[![animation of the oriented traversability maps for 32 orientations on the quarry dataset](https://i.vimeocdn.com/video/643336777_640.webp)](https://vimeo.com/224311892 "Oriented traversability maps for the quarry dataset")-->

**Pioneer 3AT on simulated procedurally generated heightmaps**

{% include vimeoPlayer.html id=224451017 %}

<!--[![video of Pioneer 3AT on simulated heightmaps](https://i.vimeocdn.com/video/643517187_640.webp)](https://vimeo.com/224451017 "Pioneer 3AT on simulated heightmaps")-->


<!--**high-quality images of the evaluation heightmaps (surfaces) and of the experiments on real robots**-->


> A larger number of visualizations can be generated/found in the `code`>`visualization` section of this repository.

## Data

This folder contains the heightmaps (gray-scale images) and the csv files with the data from each simulated trajectory.


## Code

In this folder we provide several sub-folders for each module of our traversability estimation framework:

### Simulation

Code for simulating the Pioneer 3AT robot on a generated heightmap.

This code can be accessed as a docker image that setups all the needed libraries:

> `docker pull romarcg/traversability-ros-ubuntu-gazebo`

or by copying the `simulation` folder files to an existing `ros+gazebo` setup.

### Heightmap generation

Contains a document explaining in detail the procedural generation of heightmaps and a script that implements such procedure.

Here are some examples of the generated heightmaps:

<img src="code/dataset_generation_training/heightmaps/bars1.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/bumps1.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/holes1.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/rails1.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/slope_rocks1.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/steps1.png" width="100"/>

<img src="code/dataset_generation_training/heightmaps/bars2.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/bumps2.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/holes2.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/rails2.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/slope_rocks2.png" width="100"/>
<img src="code/dataset_generation_training/heightmaps/steps2.png" width="100"/>


### Dataset generation

Contains a script that takes a csv file and builds a `dataframe` that is used to generate the training/evaluation/real-evaluation dataset.

### Training

Contains a script that builds the CNN architecture, takes the dataset generated by the previous script and starts the network training (this task may take several hours). In addition, a report of common metrics is provided as evaluation of the trained CNN.

The model generated by this script is saved to be used in future scripts.

> An already trained model file is provided to avoid the training step.

### Evaluation

This script computes oriented traversability maps and minimal traversability maps using the trained model on unseen (testing) heightmaps.

### Visualization

This script processes traversability maps and generates 3D renderings to interactively examine a heightmap and its estimated map.

> A set of already generated traversability maps is provided as sample files to avoid regenerating them.


### Software requirements

In order to use the provided scripts, these are the list of requirements:

  * python 3.5.3
  * numpy 1.12.1
  * matplotlib 2.0.0
  * pandas 0.19.2
  * tensorflow-gpu 1.0
  * keras 2.0.3
  * scikit-learn 0.18.1
  * scikit-image 0.13.0
  * joblib 0.11
  * mayavi
