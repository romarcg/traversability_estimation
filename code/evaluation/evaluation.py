#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 20 12:21:01 2017

@author: omar
"""

import numpy as np
import math

import time
import sys

import skimage
import skimage.io
import skimage.filters
import skimage.draw
import skimage.transform

import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import matplotlib.patches as patches

import pandas as pd

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import load_model

import networkx as nx

from joblib import Parallel, delayed

#%% deafult values for core variables

heightmaps_folder = "heightmaps/"    # where testing heightmaps are
model_folder = "model/"              # where the learned model is stored
output_folder = "output/"

multiprocessing = False # if True, we use 4 jobs to calculate the traversability over a full map

multiprocessing_hm = np.zeros((100,100)) # a temporal way to initialize a shared image

sim_hm_mx_x = 5.0  # heightmap dimmensions (m) used in the simulation for generating training data
sim_hm_mx_y = 5.0  # this will help to pass from sim coordinates to screen coordinates when generating datasets
                   # usually is a 10m by 10m map, so from -5 to 5

height_scale_factor = 1.0 # for learning it was 0 - 1.0, if the heightmap is higher/lower, adjust this

#%% Utility functions

def read_image(heightmap_png):
    # reads an image takint into account the scalling and the bitdepth
    hm = skimage.io.imread(heightmap_png)
    #print ("hm ndim: ",hm.ndim, "dtype: ", hm.dtype)
    if hm.ndim > 2: #multiple channels
        hm=skimage.color.rgb2gray(hm) #rgb2gray does the averaging and channel reduction
    elif hm.ndim == 2: #already in one channel
        #this is mostly for the images treated in matlab beforehand (one channel + grayscale + 16bit)
        if hm.dtype == 'uint8':
            divided = 255
        if hm.dtype == 'uint16':
            divided = 65535
        hm=hm/divided
    hm = hm * height_scale_factor #scaled to proper factor (mostly for testing, for training is 1.0)
    return hm
    

def toScreenFrame (s_x, s_y, x_max, x_min, y_max, y_min):
    # from simulation frame x right, y up, z out of the screen
    # to x right , y down, ignoring z
    xs = s_x + x_max
    ys = -s_y + y_max
    xs = xs/(x_max-x_min)
    ys = ys/(y_max-y_min)
    return xs, ys

def hmpatch(hm,x,y,alpha,edge,scale=1):   
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+edge/2, +edge/2])
    tf=(tf1+(tf2+(tf3+tf4))).inverse
    corners=tf(np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])*edge)
    patch = skimage.transform.warp(hm, tf,output_shape=(edge,edge),mode="edge")
    return patch,corners

def hmpatch_only_corners(x,y,alpha,edge,scale=1):
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+edge/2, +edge/2])
    tf=(tf1+(tf2+(tf3+tf4))).inverse
    corners=tf(np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])*edge)
    return corners

def show(patch):
    fig,ax1=plt.subplots(figsize=(7,7))
    ax1.imshow(patch,cmap="gray")#,vmin=-0.1,vmax=+0.1)
    plt.show()
    plt.close(fig)

def transform_patch(patch,sz):
    t_patch=patch-patch[patch.shape[0]//2,patch.shape[1]//2]
    t_patch = skimage.transform.resize(t_patch, (sz,sz), mode='constant')
    return t_patch    

def mc_extract_features_cnn(hm_x, hm_y, hm_g, edges, resize, scale):
    tf1 = skimage.transform.SimilarityTransform(translation=[-hm_x, -hm_y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(hm_g))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+edges/2, +edges/2])
    trf=(tf1+(tf2+(tf3+tf4))).inverse
    patch = skimage.transform.warp(multiprocessing_hm, trf,output_shape=(edges,edges),mode="edge")    
    patch = transform_patch(patch,resize) 
    return patch[:,:,np.newaxis]

def test_fitted_model_full_map_stride_cnn(fitted, heightmap_png, edges, resize, rad_ori, stride):
    print ("Generating traversability image for orientation "+ str(rad_ori) +" rads")    
    #hm=skimage.color.rgb2gray(skimage.io.imread(heightmap_png))
    #hm = hm * height_scale_factor
    hm = read_image(heightmap_png)
    X_mp=[]
    #y=[]
    hm_cols = int((np.shape(hm)[0]-edges)/stride)
    hm_rows = int((np.shape(hm)[1]-edges)/stride)
    
    full_data = pd.DataFrame()
    full_data ["patch"] = range(0,hm_cols*hm_rows)
    full_data.set_index("patch")
    print("Filling patches (by stride)")
    full_data ["hm_x"] = [ int((edges/2) + (j*stride)) for i in range(0,hm_rows) for j in range(0,hm_cols)]
    full_data ["hm_y"] = [ int((edges/2) + (i*stride)) for i in range(0,hm_rows) for j in range(0,hm_cols)]
    full_data ["G"] = [ rad_ori for i in range(0,hm_cols*hm_rows)]
    
    total_samples = len(full_data.index)
    
    startTime = time.time()
    print ("Cropping patches for feature extraction (only patch cropping)")
    if multiprocessing == False:    
        for i,d in full_data.iterrows():
            print ("\rProcessing "+ str(i) + "/" + str(total_samples), end='')
            patch=hmpatch(hm,d["hm_x"],d["hm_y"],np.rad2deg(d["G"]),edges,scale=1)[0] 
            patch=transform_patch(patch,resize) 
            X_mp.append(patch[:,:,np.newaxis])
        print ("\rProcessed "+ str(total_samples) + "/" + str(total_samples))
    else:
        print ("\rProcessing " + str(total_samples) + " [multiprocessing]", end='')
        multiprocessing_hm = read_image(heightmap_png)
        X_mp = Parallel(n_jobs=4)(delayed(mc_extract_features_cnn)(d["hm_x"],d["hm_y"],np.rad2deg(d["G"]),edges,resize,scale=1) for idx,d in full_data.iterrows())
    endTime = time.time()
    #calculate the total time it took to complete the work
    workTime =  endTime - startTime
    print ("-- time: "+ str(workTime))
    print ("Estimating traversability for all the patches")
    startTime= time.time()
    
    X = np.array(X_mp).astype('float32')
    y_pred = fitted.predict(X)
    
    endTime = time.time()
    workTime =  endTime - startTime
    print ("-- time: "+ str(workTime))
    fig,ax1=plt.subplots(figsize=(9,9))
    ax1.imshow(hm/height_scale_factor, cmap="viridis")#cmap="gray")
    fig.savefig(heightmap_png[:-4] + '_out_viridis_base' + '.png', dpi=fig.dpi)

    cax1 = ax1.imshow(hm/height_scale_factor, cmap="viridis")#cmap="gray")
    cbar = fig.colorbar(cax1, ticks=[round(np.amin(hm)+.01,2), round(np.amax(hm),2)])
    fig.savefig(heightmap_png[:-4] + '_out_viridis__bar_base' + '.png', dpi=fig.dpi)
    
    fig,ax1=plt.subplots(figsize=(9,9))
    ax1.imshow(hm/height_scale_factor, cmap="gray")
    fig.savefig(heightmap_png[:-4] + '_out_gray_base' + '.png', dpi=fig.dpi)
    
    #draw a white canvas for the traversability results
    # remove this if you want the overlay results
    #ax1.fill([0,0,np.shape(hm)[0],np.shape(hm)[0],0],[0,np.shape(hm)[1],np.shape(hm)[1],0,0],'w',alpha=1.0)
    
    # use a white skimage to draw traversabiliy results
    sk_hm = np.ones((np.shape(hm)[0],np.shape(hm)[1],4),dtype='float64') 
    sk_hm [:,:,3] =np.zeros((np.shape(hm)[0],np.shape(hm)[1]),dtype='float64')
    
    print ("Drawing predictions on patchs for current orientation")
    startTime= time.time()

    tf = skimage.transform.SimilarityTransform(translation=[edges/2,edges/2], rotation=-rad_ori)
    tf_sk = skimage.transform.SimilarityTransform(translation=[10,10], rotation=-rad_ori)
    arrow_points=tf(np.array([[0,0],[edges/2,0]]))
    arrow_points_sk=tf_sk(np.array([[-5,0],[-10,5],[5,0],[-10,-5]])) #15px arrowhead for skimage ploting    
    
    ax1.arrow(arrow_points[0][0],arrow_points[0][1],arrow_points[1][0]-arrow_points[0][0],arrow_points[1][1]-arrow_points[0][1], length_includes_head = True, width=3)
    patches_squares = []
    patches_colors = []
    for i,d in full_data.iterrows():
        print ("\rProcessing "+ str(i) + "/" + str(total_samples), end='')
        corners = hmpatch_only_corners(d["hm_x"],d["hm_y"],np.rad2deg(d["G"]),stride,scale=1)
        color_box_sk= [0.0,1.0,0.0,y_pred[i][1]] # green with prob of non-trav in alpha
        # if we only want to draw the patch without its orientations, use this (avoids holes between patches)
        s_patch = skimage.draw.polygon([ corners[4,1]-stride/2,  corners[4,1]+stride/2, corners[4,1]+stride/2, corners[4,1]-stride/2, corners[4,1]-stride/2], [ corners[4,0]-stride/2, corners[4,0]-stride/2, corners[4,0]+stride/2, corners[4,0]+stride/2, corners[4,0]-stride/2])
        skimage.draw.set_color(sk_hm, (s_patch[0],s_patch[1]), color_box_sk)
    # for ploting with skimage
    sk_hm_pure = np.copy(sk_hm)
    s_patch = skimage.draw.polygon( arrow_points_sk[[0,1,2,3,0],1], arrow_points_sk[[0,1,2,3,0],0])    
    skimage.draw.set_color(sk_hm, (s_patch[0],s_patch[1]), [0.0,0.0,1.0,1.0])
    ax1.imshow(sk_hm)
    
    #fig.savefig(heightmap_png[:-4] + '_out_' + ("%.3f" % rad_ori) + '.png', dpi=fig.dpi)
    skimage.io.imsave(heightmap_png[:-4] + '_out_' + ("%.3f" % rad_ori) + '.png' ,sk_hm_pure) #sk_hm if you want to save the arrows

    endTime = time.time()
    workTime =  endTime - startTime
    print ("\rProcessed "+ str(total_samples) + "/" + str(total_samples))
    print ("-- time: "+ str(workTime))    
    #plt.show()
    return sk_hm_pure


# generates a traversability graph using different number of neighbours (angles): 4 or 8
# Remember that gazebo frame of reference for orientation is different from vrep
# gazebo 0 rads is at the right and goes counter-clock wise
def generate_traversability_graph_cnn(fitted, heightmap_png, edges, resize, stride, neighbours):
    #
    print ("Generating traversability map/graph for " + str(neighbours)+ " orientations")
    startTime = time.time()
    
    #hm=skimage.color.rgb2gray(skimage.io.imread(heightmap_png))
    #hm = hm * height_scale_factor
    hm = read_image(heightmap_png)
    X_mp=[]
    hm_cols = int((np.shape(hm)[0]-edges)/stride)
    hm_rows = int((np.shape(hm)[1]-edges)/stride)
    
    d_nn = [[1,0],[0,-1],[-1,0],[0,1]] #position of the neighbours
    d_nn_dist = [(sim_hm_mx_x*2)/np.shape(hm)[0]*stride,(sim_hm_mx_y*2)/np.shape(hm)[1]*stride,(sim_hm_mx_x*2)/np.shape(hm)[0]*stride,(sim_hm_mx_y*2)/np.shape(hm)[1]*stride] # distance (m) between the node and its neighbours
    nn = [0, np.pi/2, np.pi, 3*np.pi/2]
    if neighbours == 8:
        d_nn = [[1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,1], [1,1]]
        nn = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        d_nn_dist = [(sim_hm_mx_x*2)/np.shape(hm)[0]*stride,
                     math.sqrt(math.pow((sim_hm_mx_y*2)/np.shape(hm)[1]*stride,2.0) + math.pow((sim_hm_mx_y*2)/np.shape(hm)[1]*stride,2.0)),
                     (sim_hm_mx_y*2)/np.shape(hm)[1]*stride,
                     math.sqrt(math.pow((sim_hm_mx_y*2)/np.shape(hm)[1]*stride,2.0) + math.pow((sim_hm_mx_y*2)/np.shape(hm)[1]*stride,2.0)),
                     (sim_hm_mx_x*2)/np.shape(hm)[0]*stride,
                     math.sqrt(math.pow((sim_hm_mx_y*2)/np.shape(hm)[1]*stride,2.0) + math.pow((sim_hm_mx_y*2)/np.shape(hm)[1]*stride,2.0)),
                     (sim_hm_mx_y*2)/np.shape(hm)[1]*stride,
                     math.sqrt(math.pow((sim_hm_mx_y*2)/np.shape(hm)[1]*stride,2.0) + math.pow((sim_hm_mx_y*2)/np.shape(hm)[1]*stride,2.0))] # distance (m) between the node and its neighbours, diagonasl are not the same length
   
    full_data = pd.DataFrame()
    full_data ["index"] = [i for j in range(len(nn)) for i in range(0,(hm_cols*hm_rows))] # range(0,(hm_cols*hm_rows)) * len(nn)
    full_data.set_index("index")
    
    full_data ["patch"] = [ i for i in range(0,(hm_cols*hm_rows)) for j in nn]
    full_data ["hm_x"] = [ int((edges/2) + (j*stride)) for i in range(0,hm_rows) for j in range(0,hm_cols) for angle in nn]
    full_data ["hm_y"] = [ int((edges/2) + (i*stride)) for i in range(0,hm_rows) for j in range(0,hm_cols) for angle in nn]
    full_data ["G"] = [ angle for i in range(0,(hm_cols*hm_rows)) for angle in nn]
    full_data ["neighbour"] = [ i_n for i in range(0,(hm_cols*hm_rows)) for i_n in range(0, len(nn))]
    
    print ("Extracting features (only patch cropping) from the height map: ")
    
    total_samples = len(full_data.index)
    if multiprocessing == False:
        for i,d in full_data.iterrows():
            print ("\rProcessing "+ str(i) + "/" + str(total_samples), end='')
            patch=hmpatch(hm,d["hm_x"],d["hm_y"],np.rad2deg(d["G"]),edges,scale=1)[0] 
            patch=transform_patch(patch,resize) 
            #features=skimage.feature.hog(skimage.transform.resize(patch,(resize_patch_size,resize_patch_size)))    
            X_mp.append(patch[:,:,np.newaxis])
        print ("\rProcessed "+ str(total_samples) + "/" + str(total_samples))
    else:
        multiprocessing_hm = read_image(heightmap_png)
        X_mp = Parallel(n_jobs=4)(delayed(mc_extract_features_cnn)(d["hm_x"],d["hm_y"],np.rad2deg(d["G"]),edges,resize ,scale=1) for idx,d in full_data.iterrows())
    print ('Estimating traversability for all the patches')
    X=np.array(X_mp).astype('float32')
    
    y_pred = fitted.predict(X) # it returns an array with probabilites per class per sample
    
    fig,ax1=plt.subplots(figsize=(7,7))
    ax1.imshow(hm,cmap="gray")
    
    G = nx.DiGraph()
    G.add_nodes_from(np.unique(full_data["patch"]))
    g_labels = dict(zip(np.unique(full_data["patch"]),np.unique(full_data["patch"])))
    nx.set_node_attributes(G,"label",g_labels)
    #i = 0
    print ("Generating traversability graph per patch:")
    patches_centers = []
    patches_tra_angles_lines = []
    for index, d in full_data.iterrows():
        print ("\rProcessing "+ str(index) + "/" + str(total_samples), end='')
        if True: #y_pred[index][1]>0.5: # prob of False is in index 0, True in index 1, only for traversable patches
            conn_neighbour = int(d["patch"]) + (d_nn[int(d["neighbour"])][1] * hm_cols) + (d_nn[int(d["neighbour"])][0])
            valid_edge = False
            if conn_neighbour >=0 and conn_neighbour < (hm_cols*hm_rows): # no negative vertices or more than the existing ones
                if d["patch"]%hm_cols !=0 and (d["patch"]+1)%hm_cols !=0:  #if patch is not in the leftest edge of the map
                    valid_edge = True
                elif (d["patch"]%hm_cols == 0) and (d["G"] <= 1.58 or d["G"] >= 4.71):
                    valid_edge = True
                elif ((d["patch"]+1)%hm_cols == 0) and (d["G"] >= 1.57 and d["G"] <= 4.72):
                    valid_edge = True
            if valid_edge: # add edge to the graph
                G.add_edge(int(d["patch"]), int(conn_neighbour), probability=y_pred[index][1], weight=(1-y_pred[index][1]), label=round(y_pred[index][1],1), neighbour = conn_neighbour, distance=d_nn_dist[int(d["neighbour"])])
                if y_pred[index][1]<=0.5: #only draw the non-traversable edges in the plot to avoid overdrawing overlays
                    patches_centers.append([d["hm_x"], d["hm_y"]])                
                    patches_tra_angles_lines.append([(d["hm_x"], d["hm_y"]), (d["hm_x"]+ (d_nn[int(d["neighbour"])][0] * int(0.9*stride/2)), d["hm_y"]+ (d_nn[int(d["neighbour"])][1] * int(0.9*stride/2)))])
    patches_centers = np.array(patches_centers)
    ax1.scatter(patches_centers[:,0], patches_centers[:,1], s=1, c=(1,0,1,0.2))
    patches_tra_angles_lines = np.array(patches_tra_angles_lines)
    lc = mc.LineCollection(patches_tra_angles_lines, colors=(1,0,0,0.5), linewidths=1)
    ax1.add_collection(lc)
    print ("\rProcessed "+ str(total_samples) + "/" + str(total_samples))
    
    # to save the graph in pygraphviz for visualization
    nodes_positions_gviz = [ str(j)+','+str(-i)+'!' for i in range(0, hm_rows) for j in range(0,hm_cols)]
    #attr_positions = dict (zip(np.unique(full_data["patch"]),nodes_positions))
    attr_positions = dict (zip(np.unique(full_data["patch"]),nodes_positions_gviz))
    nx.set_node_attributes(G,"pos",attr_positions)
    ##nx.draw(G,nx.get_node_attributes(G,'pos'))
    #A=nx.drawing.nx_agraph.to_agraph(G)
    #A.write('t_graph_cnn.dot') 
    nx.write_gpickle(G, "t_graph_cnn.gpickle")
    
    # to visualize the graph in graphviz (and forcing to use our position for the nodes), use: 
    # dot t_graph.dot -Tx11 -Kneato
    
    # to read a dot file we shlould use:
    #G = nx.drawing.nx_agraph.read_dot('t_graph_cnn.dot')
    
    endTime = time.time()
    # compute the total time it took to complete the work
    workTime =  endTime - startTime    
    print ("Traversability graph generation time: "+ str(workTime))
    
    return full_data, y_pred, G


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Setup for evaluation on specific heigthmaps
#

#
# load fitted model from keras
#
model = keras.models.load_model(model_folder+"traversability_pioneer_b150_spe_100_e_50_acc_82.h5")


# Note:
# For each heighmap you want to test, set the corresponding variables that are 
# needed to do so

# Here are some examples:
    
#% for the quarry 2cm rescaled
heightmap_png = heightmaps_folder+ "quarry_cropped4_scaled2cm.png" 
patch_size = 60
patch_resize = 60
stride = 9
sim_hm_mx_x = 15.2
sim_hm_mx_y = 15.2
height_scale_factor = 10.0

#% for a custom artificial map
# open the custom artificial scene in vrep heightmap1
heightmap_png = heightmaps_folder+"custom9.png"
patch_size = 60
patch_resize = 60
stride = 7
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 1.0

#% for another custom artificial map
heightmap_png = heightmaps_folder+"custom2.png"
patch_size = 60
patch_resize = 60
stride = 5
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 1.0

#% for another custom artificial map
heightmap_png = heightmaps_folder+"heightmap1.png"
patch_size = 60
patch_resize = 60
stride = 5
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 1.0

#% for another custom artificial map
heightmap_png = heightmaps_folder+"custom13.png"
patch_size = 60
patch_resize = 60
stride = 3
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 1.0


#% for another custom artificial map
heightmap_png = heightmaps_folder+"custom1.png"
patch_size = 60
patch_resize = 60
stride = 3
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 1.0

#% for uzh heightmap, 
heightmap_png = heightmaps_folder+"uzh_elevation.png"
patch_size = 12
patch_resize = 60
stride = 1
sim_hm_mx_x = 9.1# for the custom quarry map
sim_hm_mx_y = 9.1
height_scale_factor = 5.0


# for the realworld evaluation datasets (gravelpit)
heightmap_png = heightmaps_folder+"gravelpit1.png"
patch_size = 60
patch_resize = 60
stride = 3
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 3.0

# for the realworld evaluation datasets (gravelpit)
heightmap_png = heightmaps_folder+"gravelpit5.png"
patch_size = 60
patch_resize = 60
stride = 3
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 3.0

# for the dataset with rocks in an arc shape
heightmap_png = heightmaps_folder+"arc_rocks.png"
patch_size = 60
patch_resize = 60
stride = 5
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 0.4


#% for ETH in ASL's lab heightmap (re scale version with 2cm assumption)
heightmap_png = heightmaps_folder+"gridmap_elevation_2_c_r.png"
patch_size = 60
patch_resize = 60
stride = 5
sim_hm_mx_x = 4.75
sim_hm_mx_y = 4.75
height_scale_factor = 0.739174


#% for a custom artificial map
heightmap_png = heightmaps_folder+"bars_6_9_14_18.png"
patch_size = 40
patch_resize = 40
stride = 3
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 0.4


#
# set these lines for multiprocessing 
#

multiprocessing = True
multiprocessing_hm = read_image(heightmap_png)

# evaluate for specific orientations  : outputs are inside heightmaps folder
#
temp_thm = test_fitted_model_full_map_stride_cnn(model,heightmap_png, patch_size, patch_resize, 0, stride)
temp_thm = test_fitted_model_full_map_stride_cnn(model,heightmap_png, patch_size, patch_resize, np.pi/2, stride)
temp_thm = test_fitted_model_full_map_stride_cnn(model,heightmap_png, patch_size, patch_resize, np.pi, stride)
temp_thm = test_fitted_model_full_map_stride_cnn(model,heightmap_png, patch_size, patch_resize, 3*np.pi/2, stride)

# evaluate for a set of orientations (in this case 32) : outputs are inside heightmaps folder
#
test_oris = 32
test_ori_step = 2*np.pi/test_oris
for i in range (0,test_oris):
    temp_thm = test_fitted_model_full_map_stride_cnn(model,heightmap_png, patch_size, patch_resize, i*test_ori_step, stride)

full_data, y_pred, G = generate_traversability_graph_cnn(model, heightmap_png, patch_size, patch_resize, stride, 8)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# mayavi functions for generating interactive visualization and the minimal traversability map
#
import os
os.environ["QT_API"] = "pyqt"

from mayavi import mlab
from tvtk.api import tvtk

from traits.api import HasTraits, Range, Instance, on_trait_change
from traitsui.api import View, Item, HGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene


class Visualization(HasTraits):
    orientation = Range(0,32,0)
    scene      = Instance(MlabSceneModel, ())    
    vector_traversabilty_files = []
    vector_traversabilty_color_files = []

    def __init__(self, heightmap_png, dir_path, resolution, height_scale_factor):
        HasTraits.__init__(self)
        self.heightmap_png =heightmap_png
        self.dir_path=dir_path
        self.resolution=resolution
        self.height_scale_factor=height_scale_factor
        
        self.hm = self.read_image_hm(self.heightmap_png)
        
        self.retrieve_vector_traversabilty_from_files(self.dir_path)
        self.max_n_orientations = np.shape(self.vector_traversabilty_files)[0]
        
        self.y, self.x = np.meshgrid(np.arange(self.hm.shape[0])*resolution,np.arange(self.hm.shape[1])*resolution)
        self.surface_hm = self.scene.mlab.surf(self.x, self.y, self.hm, color=(0.8,0.8,0.8), figure=self.scene.mayavi_scene)#, warp_scale="auto")
        
        mask_t = self.retrieve_colormask_traversability(0, True)
        mask_f = self.retrieve_colormask_traversability(0, False)

        self.surface_hm_t = self.scene.mlab.surf(self.x, self.y, self.hm+0.05, mask=mask_t, color=(0.6,0.8,0.6), figure=self.scene.mayavi_scene)#, warp_scale="auto")

        self.surface_hm_f = self.scene.mlab.surf(self.x, self.y, self.hm+0.05, mask=mask_f, color=(0.8,0.6,0.6), figure=self.scene.mayavi_scene)#, warp_scale="auto")

    def read_image_hm(self,heightmap_png):
        # reads an image takint into account the scalling and the bitdepth
        hm = skimage.io.imread(heightmap_png)
        #print ("hm ndim: ",hm.ndim, "dtype: ", hm.dtype)
        if hm.ndim > 2: #multiple channels
            hm=skimage.color.rgb2gray(hm) #rgb2gray does the averaging and channel reduction
        elif hm.ndim == 2: #already in one channel
            #this is mostly for the images treated in matlab beforehand (one channel + grayscale + 16bit)
            if hm.dtype == 'uint8':
                divided = 255
            if hm.dtype == 'uint16':
                divided = 65535
            hm=hm/divided
        hm = hm * self.height_scale_factor #scaled to proper factor (mostly for testing, for training is 1.0)
        return hm

    def read_image_with_traversability(self, heightmap_png):
        # reads an image generated by our traversability estimation
        hm = skimage.io.imread(heightmap_png)/255.0
        return hm
    
    def retrieve_vector_traversabilty_from_files(self, dir_path):
        #remember to read only pngs of the same size as the hm
        self.vector_traversabilty_files = []
        self.vector_traversabilty_color_files = []
        list_files = os.listdir(dir_path)
        list_files.sort()
        for filename in list_files:        
            t_e_hm = self.read_image_with_traversability(dir_path+filename)
            mask_t=t_e_hm[:,:,0]>0.5
            mask_f=t_e_hm[:,:,1]>0.5
            self.vector_traversabilty_files.append([mask_t,mask_f])
            self.vector_traversabilty_color_files.append(t_e_hm)
    
    def traversability_color_mappping(self,idx_orientation):
        # builds a color vector wit traversability info (this is another way 
        # to color the surface without using masks (time consumming to render))
        mask_t = self.retrieve_colormask_traversability(idx_orientation, True)
        mask_f = self.retrieve_colormask_traversability(idx_orientation, False)
        colors = np.zeros((mask_t.shape[0]*mask_t.shape[1]))
        for i in range(mask_t.shape[0]):
            for j in range(mask_t.shape[1]):
                if (mask_t[i,j] == True and (mask_t[i,j] == mask_f[i,j])):
                    colors[i*mask_t.shape[0] + j] = 0.1
                elif (mask_t[i,j] == True):
                    colors[i*mask_t.shape[0] + j] = 0.8
                else:
                    colors[i*mask_t.shape[0] + j] = 0.5
        return colors
    
    def retrieve_colormask_traversability(self, idx_orientation, istraversable):
        t_e_hm = self.vector_traversabilty_files[idx_orientation]
        mask=[]
        if istraversable == True:
            mask=t_e_hm[0]
        else:
            mask=t_e_hm[1]
        return mask


    def compute_compose_min_traversability(self):
        # computes the minimal traversability map from oriented traversability maps
        # that were generated by function test_fitted_model_full_map_stride_cnn
        compose_t = None
        for shm in self.vector_traversabilty_color_files:            
            if compose_t == None: #first time, create image
                compose_t = np.ones((np.shape(shm)[0],np.shape(shm)[1]),dtype='float64') 
                compose_t.fill(-1)
                for i in range(0,shm.shape[0]):
                    for j in range(0,shm.shape[1]):
                        if (shm[i,j,0] == 1.0 and shm[i,j,1] == 1.0 and shm[i,j,2] == 1.0): #margins are  white with 0 alpha
                            compose_t[i,j] = -1
                        else: # traversability is stored in alpha of each estimated image for x orientation
                            compose_t[i,j] = shm[i,j,3]
            else: 
                for i in range(0,shm.shape[0]):
                    for j in range(0,shm.shape[1]):
                        if (shm[i,j,0] == 1.0 and shm[i,j,1] == 1.0 and shm[i,j,2] == 1.0): #margins
                            compose_t[i,j] = compose_t[i,j]
                        else:  # traversability is stored in alpha of each estimated image for x orientation
                            compose_t[i,j] = min(compose_t[i,j], shm[i,j,3])
        return compose_t
        

    @on_trait_change('orientation')
    def update_plot(self):
        if self.orientation < self.max_n_orientations:
            #print ('current orientation idx:',self.orientation)
            mask_t = self.retrieve_colormask_traversability(self.orientation, True)
            mask_f = self.retrieve_colormask_traversability(self.orientation, False)
                        
            self.surface_hm_t.mlab_source.set(mask = mask_t, scalars = self.hm+0.05)
            self.surface_hm_f.mlab_source.set(mask = mask_f, scalars = self.hm+0.05)
        else:
            self.orientation = self.max_n_orientations-1


    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    #height=250, width=300, 
                    show_label=False),
                HGroup('_', 'orientation', ), resizable=True
                )

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# visualizing wiith mayavi (interactive mode for several orientations)
#

#
# First setup some needed variables for a specfic heighmap
#

# for the quarry
heightmap_png =  heightmaps_folder+'quarry_cropped4_scaled2cm.png'
hm_name_noext = 'quarry_cropped4_scaled2cm'
dir_computed_traversability_files = output_folder + hm_name_noext+'/animation_mayavi/'
resolution_hm=0.02
height_scale_factor_hm=10.0

# for a custom heightmap
heightmap_png =  heightmaps_folder+'custom2.png'
hm_name_noext = 'custom2'
dir_computed_traversability_files = output_folder + hm_name_noext+'/animation_mayavi/'
resolution_hm=0.02
height_scale_factor_hm=2.0

# for uzh
heightmap_png =  heightmaps_folder+'uzh_elevation_smoothed.png'
hm_name_noext = 'uzh_elevation_smoothed'
dir_computed_traversability_files = output_folder + hm_name_noext+'/animation_mayavi/'
resolution_hm=0.12
height_scale_factor_hm=5.0


# for the arc rocks
heightmap_png =  heightmaps_folder+'arc_rocks.png'
hm_name_noext = 'arc_rocks'
dir_computed_traversability_files = output_folder + hm_name_noext+'/animation_mayavi/'
resolution_hm=0.02
height_scale_factor_hm=0.4

# for the eth als lab heightmap rescaled
heightmap_png =  heightmaps_folder+'gridmap_elevation_2_c_r.png'
hm_name_noext = 'gridmap_elevation_2_c_r'
dir_computed_traversability_files = output_folder + hm_name_noext+'/animation_mayavi/'
resolution_hm=0.02
height_scale_factor_hm=0.739174


visualization = Visualization(heightmap_png, dir_computed_traversability_files, resolution_hm, height_scale_factor_hm)
# uncomment this two lines if you want to open an interactive view of the 
# traversability maps renderings (in may take few mins to render and navigate)
#

#visualization.configure_traits()
#visualization.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

#
# Here we compute the minimal traversaility map and save the image
#
comp_t = visualization.compute_compose_min_traversability()
skimage.io.imsave("min_traversability_map_gray.png", comp_t)
color_comp_t = np.zeros((np.shape(comp_t)[0],np.shape(comp_t)[1],4),dtype='float64') 
color_comp_t[:,:,1].fill(1)
color_comp_t[:,:,3] = comp_t
skimage.io.imsave("min_traversability_map_color.png", color_comp_t)

