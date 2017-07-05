# Code

Source code is divided in 5 sections:

1. `simulation` contains scripts and packages needed to:
  * setup the simulated world
  * simulate the pioneer 3AT on a generated hegithmap 
  * run experiments and store the data on csv files

> the content of this folder is already setup in the docker image: [link](#link)

2. `heightmap_generation` contains scripts for generating elevation maps with real terrain patterns

3. `dataset_generation_training` contains a script that takes the csv files and heightmaps, and generate the training and evaluation datasets

4. `dataset_generation_training` contains a script that defines de CNN architecture and train the classifier using the generated dataset. This script generates and store a model file. Also, at the end of the script classifier evaluation metrics (stats) are computed.

> this process may take hours depending on the machine capabilities

> an already trained model is provided as a file named `traversability_pioneer_b150_spe_100_e_50_acc_82.h5`

5. `evaluation` contains a script that loads the learned model and compute oriented traversability maps, minimal traversability maps using oriented traversability maps, and traversability graphs used for planning.

4. `visualization` contains script to visualize traversability representations (such as traversability maps or min traversability overlays) as 3D renderings on testing heightmaps (e.g. mining quarry)

> The mining quarry heightmap is included as an example in all the scripts for quick setup and testing
