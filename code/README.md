# Code

Source code is divided in 5 sections:

1. `simulation` contains scripts and packages needed to:
  * setup the simulated world
  * simulate the pioneer 3AT on a generated hegithmap 
  * run experiments and store the data on csv files

> the content of this folder is already setup in the docker image: [link](#link)

2. `heightmap_generation` contains scripts for generating elevation maps with real terrain patterns

3. `dataset_generation` contains scripts that take the csv files and heightmaps, and generate the training and evaluation datasets

4. `training` contains a script that define de CNN architecture and train the classifier using the generated dataset. This script generates and store a model file.

> this process may take hours depending on the machine capabilities

> an already trained model is provided as a file named `traversability_pioneer_b150_spe_100_e_50_acc_82.h5`

5. `evaluation` contains a script that loads the learned model and compute evaluation stats 

4. `visualization` contains script to calculate and visualize traversability representations (such as traversability maps or min traversability overlays) on testing heightmaps (e.g. mining quarry)
