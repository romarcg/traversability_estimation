#Evaluation of the traversability learned model from dataset_generation_training script

This scripts loads a trained model `.h5` file and estimates the traversability over new heightmaps. As a result it outputs:
- oriented traversability maps as green overlay images indicating traversability for the particular orientation
- minimal traversability maps from oriented traversability maps
- traversability graphs for planning

The outputs of this script are used as inputs for the `visualization` script that will generate 3D renderings, interactive surfaces and animations for the respective inputs.
