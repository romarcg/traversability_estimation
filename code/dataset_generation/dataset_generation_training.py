#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 10 12:21:01 2017

@author: omar
"""

# pandas
import pandas as pd

# skimage and plotting
import skimage
import skimage.io
import skimage.feature
import skimage.novice
import matplotlib.pyplot as plt

# numpy
import numpy as np
import math

# skelearn
import sklearn.pipeline
import sklearn.dummy
import sklearn.preprocessing
import sklearn.metrics.regression
from sklearn.metrics import auc, roc_curve
import skimage.transform

# keras and tensorflow
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import load_model, save_model

# utility
import time

# graphs
import networkx as nx

# multicore processing 
from joblib import Parallel, delayed

#%% deafult values for core variables

# Note: default values are for the sample csv/heightmaps files provided in the repository

csvs_folder = "csvs/"
heightmaps_folder = "heightmaps/"
output_folder = "output/"
training_meta_file= "meta_training.csv"
evaluation_meta_file= "meta_evaluation.csv"

evaluation_meta_file_real= "meta_evaluation_real.csv" #for the realworld dataset (i.e. gravelpit)
                   #
                   # Each csv contains 20s of simulation from gazebo: pioneer on heightmap
time_window = 75   # Ideally: each row in the csv is about 0.01 segs, so 50 is about .50 s in the future, 100 1. s
                   # However: sometimes due to the simulation load, each row is about 0.02 segs. 
                   # A preferred way of counting would be in time instead of windows.

                #
patch_size = 60 # for extracting patchs from the heightmap for training, eval and testing datasets
                # Pioneer is about 50cm long x 47cm wide x 40cm height
                # heightmaps are render with a 2cm per pixel resolution; 513x513px --> 10x10m ~1m maxheight
                
patch_size_training = patch_size # with 30 we resize to this for training the cnn, it will allow us to deal with small maps

                      #
advancement_th = 0.10 # threshold in meters use to generate the training dataset, i.e. when a patch is traversed
                      # this has to be set according to the pioneer velocity and its ideal displacement (flat land)
                      # .15m/s is the current linear velocity (assuming a forward no steering control)
                      # ergo, ideal_displacement = .15m/s x (timewindow in seconds)

           #
debug = 0  # debug level for extra logging and intermedia plots, 0 no debuggin -- 3 shows everyhing
           #

multiprocessing = False # if True, we use jobs to generate dataset/calculate the traversability/ plot over a full map 

multiprocessing_hm = np.zeros((100,100)) # a temporal way to initialize a shared image

sim_hm_mx_x = 5.0  # heightmap dimmensions (m) used in the simulation for generating training data
sim_hm_mx_y = 5.0  # this will help to pass from sim coordinates to screen coordinates when generating datasets
                   # usually is a 10m by 10m map, so from -5 to 5


height_scale_factor = 1.0 # for learning heightmaps it was 0 - 1.0; if heighmaps are higher, change accordingly 

#%% utility functions

#read and scale the heightmap elevation
def read_image(heightmap_png):
    # reads an image takint into account the scalling and the bitdepth
    hm = skimage.io.imread(heightmap_png)
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
    
# transform from simulation frame to image frame
def toScreenFrame (s_x, s_y, x_max, x_min, y_max, y_min):
    # from simulation frame x right, y up, z out of the screen
    # to x right , y down, ignoring z
    xs = s_x + x_max
    ys = -s_y + y_max
    xs = xs/(x_max-x_min)
    ys = ys/(y_max-y_min)
    return xs, ys

# extract heightmap patch

def hmpatch(hm,x,y,alpha,edge,scale=1):
    # Cutout a patch from the image, centered on (x,y), rotated by alpha
    # degrees (0 means bottom in hm remains bottom in patch, 90 means bottom in hm becomes right in patch),
    # with a specified edge size (in pixels) and scale (relative).    
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+edge/2, +edge/2])
    tf=(tf1+(tf2+(tf3+tf4))).inverse
    #corners=tf(np.array([[0,0],[1,0],[1,1],[0,1]])*edge)
    corners=tf(np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])*edge)
    patch = skimage.transform.warp(hm, tf,output_shape=(edge,edge),mode="edge")
    return patch,corners

def hmpatch_only_corners(x,y,alpha,edge,scale=1):
    # Cutout a patch from the image, centered on (x,y), rotated by alpha
    # degrees (0 means bottom in hm remains bottom in patch, 90 means bottom in hm becomes right in patch),
    # with a specified edge size (in pixels) and scale (relative).
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+edge/2, +edge/2])
    tf=(tf1+(tf2+(tf3+tf4))).inverse
    corners=tf(np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])*edge)
    #patch = skimage.transform.warp(hm, tf,output_shape=(edge,edge),mode="edge")
    return corners


# Show the 20 slowest and the 20 fastest patches
def show(sample,hm):
    patch=hmpatch(hm,sample["hm_x"],sample["hm_y"],np.rad2deg(sample["S_RCO_G"]),patch_size,scale=1)[0] # make sure to extract the patch from the correct heightmap
    patch=patch-patch[patch.shape[0]//2,patch.shape[1]//2]
    fig,ax1=plt.subplots(figsize=(7,7))
    ax1.imshow(patch/height_scale_factor,cmap="coolwarm",vmin=-0.1,vmax=+0.1)
    ax1.set_title("advancement: {:.4f}".format(sample["advancement"]))
    plt.show()
    plt.close(fig)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Dataset generation fucntions
#

def transform_patch(patch,sz):
    t_patch=patch-patch[patch.shape[0]//2,patch.shape[1]//2]
    t_patch = skimage.transform.resize(t_patch, (patch_size_training,patch_size_training), mode='constant')
    return t_patch    

def generate_single_dataset_cnn(input_csv, heightmap_png):
    hm = read_image(heightmaps_folder+heightmap_png)
    df=pd.read_csv(csvs_folder+input_csv).set_index("TIMESTAMP")
    df.columns=df.columns.map(str.strip) # strip spaces
    if debug>1:
        plt.figure();
        df.plot.scatter(x="S_RCP_X", y="S_RCP_Y")
    
    #% Convert xy to hm coords 
    df["hm_x"]=df.apply(lambda r: toScreenFrame(r["S_RCP_X"], r["S_RCP_Y"], sim_hm_mx_x, -sim_hm_mx_x, sim_hm_mx_y, -sim_hm_mx_y)[0]*hm.shape[1], axis=1)
    df["hm_y"]=df.apply(lambda r: toScreenFrame(r["S_RCP_X"], r["S_RCP_Y"], sim_hm_mx_x, -sim_hm_mx_x, sim_hm_mx_y, -sim_hm_mx_y)[1]*hm.shape[0], axis=1)
    
    
    #% Plot trajectory
    if debug>0:    
        fig,ax=plt.subplots(figsize=(15,15))
        ax.imshow(hm/height_scale_factor)
        ax.plot(df["hm_x"], df["hm_y"], '-y')
        ax.plot(df["hm_x"].iloc[0], df["hm_y"].iloc[0], 'oy')
    
    #% Plot angles
    #import numpy as np
    if debug>1:
        plt.figure();
        np.rad2deg(df["S_RCO_G"]).plot()
    
    #%
    # Unit vector of robot orientation
    df["S_oX"]= np.cos(df["S_RCO_G"])
    df["S_oY"]= np.sin(df["S_RCO_G"])
    assert(np.allclose(1,np.linalg.norm(df[["S_oX","S_oY"]],axis=1)))
    
    # dX, dY, distance at 10 timesteps in the future
    dt=time_window
    df["S_dX"]=df.rolling(window=(dt+1))["S_RCP_X"].apply(lambda x: x[-1]-x[0]).shift(-dt)
    df["S_dY"]=df.rolling(window=(dt+1))["S_RCP_Y"].apply(lambda x: x[-1]-x[0]).shift(-dt)
    df["S_d"]=np.linalg.norm(df[["S_dX","S_dY"]],axis=1)
    if debug>1:
        plt.figure();
        df["S_d"].plot()
    
    #% Project dX, dY on current direction
    df["advancement"]=np.einsum('ij,ij->i', df[["S_dX","S_dY"]], df[["S_oX","S_oY"]]) # row-wise dot product   
    
    # set the label using a threshold value
    df["label"]=df["advancement"]>advancement_th
               
    #% Filter data
    # skip the first two seconds and any row with nans (i.e. end of the dataset)
    dff=df.loc[df.index>=2].dropna()
    dff=dff.loc[dff.index<=18].dropna() # drop also the last two seconds (if run is 20s, < 18)
    
    # drop the frames where the robot is upside down (orientation alpha angle [euler's angles]) to avoid false positives
    dff=dff.loc[dff.S_RCO_A>=-2.0].dropna() 
    dff=dff.loc[dff.S_RCO_A<=2.0].dropna() 

    dff=dff.loc[dff.S_RCO_B>=-2.0].dropna()
    dff=dff.loc[dff.S_RCO_B<=2.0].dropna()
    
    #% Visualize the data
    if debug>2:
        print("Slow")
        for i,sample in dff.sort_values("advancement").head(20).iterrows():
            show(sample,hm)
        
        print("Fast")    
        for i,sample in dff.sort_values("advancement").tail(20).iterrows():
            show(sample,hm)
    
    return dff, hm

# choose a single csv and select some of its patches (for visualization and evaluation)
def make_charrypicked_dataset(df,hm,_skips):
    X=[]
    y=[]
    skips = _skips
    i=0
    #cp_df = pd.DataFrame(columns=df.columns.values)
    #new_i = 0
    row_list = []
    patches_img = []
    for idx,d in df.iterrows():
        if (i%skips) == 0:
            # make sure to extract the patch from the correct heightmap
            patch=hmpatch(hm,d["hm_x"],d["hm_y"],np.rad2deg(d["S_RCO_G"]),patch_size,scale=1)[0] 
            patch=patch-patch[patch.shape[0]//2,patch.shape[1]//2]                
            X.append(patch[:,:,np.newaxis])
            y.append(d["advancement"]>advancement_th) # define the class here
            row_list.append(d)
        i = i+1
    cp_df = pd.DataFrame(row_list)  
    X=np.array(X).astype('float32')
    y=np.array(y)
    y=keras.utils.to_categorical(y,2)
    return X,y, cp_df

def cherrypick_single_dataset (input_csv, heightmap_png, _skips = 1):
    df, hm = generate_single_dataset_cnn(input_csv, heightmap_png)
    X,y,cp_df = make_charrypicked_dataset(df,hm, _skips)
    plt.figure();
    fig,ax1=plt.subplots(figsize=(7,7))
    ax1.imshow(hm,cmap="gray")
    fig2,ax21=plt.subplots(figsize=(7,7))
    i = 0
    sk_hm = np.ones((np.shape(hm)[0],np.shape(hm)[1]),dtype='float64') 
    sk_hm = skimage.color.gray2rgb(sk_hm)
    for idx,d in cp_df.iterrows():
        # make sure to extract the patch from the correct heightmap
        patch,corners=hmpatch(hm,d["hm_x"],d["hm_y"],np.rad2deg(d["S_RCO_G"]),patch_size,scale=1)
        color_box_sk= [1.0,0.0,0.0] #red
        if y[i][1]>0.5:
            color_box_sk= [0.0,1.0,0.0] #geen
            ax1.fill(corners[[0,1,2,3,0],0],corners[[0,1,2,3,0],1],'g',alpha=0.2)
            ax1.plot(corners[[0,1,2,3,0],0],corners[[0,1,2,3,0],1],'g-',alpha=0.3)
            skimage.io.imsave(heightmap_png[:-4] + '_out_patch_' + str(i) + 'true' + '.png' ,(patch*255).astype(dtype=np.uint8))
            ax21.imshow(patch,cmap="viridis")
        else:
            ax1.fill(corners[[0,1,2,3,0],0],corners[[0,1,2,3,0],1],'r',alpha=0.2)
            ax1.plot(corners[[0,1,2,3,0],0],corners[[0,1,2,3,0],1],'r-',alpha=0.3)
            #print (patch)
            skimage.io.imsave(heightmap_png[:-4] + '_out_patch_' + str(i) + 'false' + '.png' ,(patch*255).astype(dtype=np.uint8))
            ax21.imshow(patch,cmap="viridis")
        s_patch = skimage.draw.polygon(corners[[0,1,2,3,0],1],corners[[0,1,2,3,0],0])
        skimage.draw.set_color(sk_hm, (s_patch[0],s_patch[1]), color_box_sk, alpha = 1.0)
        
        ax1.plot(corners[[1,2],0],corners[[1,2],1],'y-',alpha=0.7)
        i = i+1
    skimage.io.imsave(heightmap_png[:-4] + '_out_cherrypicked_' + '.png' ,sk_hm)
    fig.savefig(heightmap_png[:-4] + '_out_single_dataset_vis.png', dpi=fig.dpi)
    #print (cp_df)
    return cp_df,hm,X,y


def generate_composite_dataset_cnn(meta_csv):
    meta_file=pd.read_csv(csvs_folder+meta_csv).set_index("ID")
    meta_dataset= pd.DataFrame()
    heightmaps = dict ()
    for i,r in meta_file.iterrows():
        print ("Processing: ",r["CSV"])
        dff, hm = generate_single_dataset_cnn(r["CSV"],r["HEIGHTMAP"])
        dff["hm"] = r["HEIGHTMAP"]
        meta_dataset=meta_dataset.append(dff)
        if not (r["HEIGHTMAP"] in heightmaps):
            heightmaps[r["HEIGHTMAP"]] = hm
    return meta_dataset, heightmaps


def sample(df,hms,sz):
    d=df.sample(1)
    l=d["label"].iloc[0]
    hm = hms[d["hm"].iloc[0]]
    patch=hmpatch(hm,d["hm_x"].iloc[0],d["hm_y"].iloc[0],np.rad2deg(d["S_RCO_G"].iloc[0]),patch_size,scale=1)[0]
    patch = transform_patch(patch, sz)
    return patch,l

def mktrte(df,hms,N,sz):
    X = []
    y = []
    for i in range(N):
        im,l=sample(df,hms,sz)
        X.append(im[:,:,np.newaxis])
        y.append(l)
    X=np.array(X).astype('float32')
    y=np.array(y)
    y=keras.utils.to_categorical(y,2)
    return X,y


def generator_tr(df,hms,batch_size,sz):
    while True:
        X,y = mktrte(df,hms,batch_size,sz)
        yield (X, y)

def generator_ev(df,hms,batch_size,sz):
    while True:
        X,y = mktrte(df,hms,batch_size,sz)
        yield (X, y)

#%% Functions for CNN fitting

def sklearnAUC(test_labels,test_prediction):
    K.print_tensor(test_labels)
    print (K.is_keras_tensor(test_labels),K.shape(test_labels))
    K.print_tensor(test_prediction)
    print (K.is_keras_tensor(test_prediction),K.shape(test_prediction))
    return K.mean(test_prediction) 


class CustomCallbacks(keras.callbacks.Callback): #create a custom History callback
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.params['metrics'].append('val_auc')

    def on_train_end(self, logs={}):
        return
    
    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0])
        auc_epoch = sklearn.metrics.roc_auc_score(self.validation_data[1][0], y_pred)
        logs['val_auc'] = auc_epoch
        self.aucs.append(auc_epoch)
        return
    def on_batch_begin(self, batch, logs={}):
        return
    
    def on_batch_end(self, batch, logs={}):
        return


custom_callbacks = CustomCallbacks()
#%% functions for testing the fitted cnn

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
    #
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
            #features=skimage.feature.hog(skimage.transform.resize(patch,(resize_patch_size,resize_patch_size)))    
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
    #plt.show(block=False)
    ax1.imshow(hm/height_scale_factor, cmap="rainbow")#cmap="gray")
    fig.savefig(heightmap_png[:-4] + '_out_rainbow_base' + '.png', dpi=fig.dpi)

    cax1 = ax1.imshow(hm/height_scale_factor, cmap="rainbow")#cmap="gray")
    cbar = fig.colorbar(cax1, ticks=[round(np.amin(hm)+.01,2), round(np.amax(hm),2)])
    fig.savefig(heightmap_png[:-4] + '_out_rainbow__bar_base' + '.png', dpi=fig.dpi)
    
    fig,ax1=plt.subplots(figsize=(9,9))
    ax1.imshow(hm/height_scale_factor, cmap="gray")
    fig.savefig(heightmap_png[:-4] + '_out_gray_base' + '.png', dpi=fig.dpi)
        
    # use a white skimage to draw traversabiliy results
    sk_hm = np.ones((np.shape(hm)[0],np.shape(hm)[1]),dtype='float64') 
    sk_hm = skimage.color.gray2rgb(sk_hm)
    
    #to store only true traversability values (for using later in a visualization)
    true_t_hm = np.zeros((np.shape(hm)[0],np.shape(hm)[1]),dtype='float64') 
    
    print ("Drawing predictions on patchs for current orientation")
    startTime= time.time()

    '''
    Description of SimilarityTransform states that the rotaiton angle is 
    counter-clockwise, however it does not behave like so. Be aware of this 
    issue and consider Angles from dataset are in radians and counter-clockwise 
    so PI/2 means left orientation, while 3PI/2 means right orientation, if we 
    do not invert the orientation,  skimiage transformation does the opposite. 
    
    gazebo orientation frame is different from vrep (sic), 0 degrees is at the 
    right and goes up counter-clockwise
    '''
    tf = skimage.transform.SimilarityTransform(translation=[edges/2,edges/2], rotation=-rad_ori)
    tf_sk = skimage.transform.SimilarityTransform(translation=[10,10], rotation=-rad_ori)
    arrow_points=tf(np.array([[0,0],[edges/2,0]]))
    arrow_points_sk=tf_sk(np.array([[-5,0],[-10,5],[5,0],[-10,-5]])) #15px arrowhead for skimage ploting
    # plot arrow_points
    ax1.arrow(arrow_points[0][0],arrow_points[0][1],arrow_points[1][0]-arrow_points[0][0],arrow_points[1][1]-arrow_points[0][1], length_includes_head = True, width=3)
    patches_squares = []
    patches_colors = []
    for i,d in full_data.iterrows():
        print ("\rProcessing "+ str(i) + "/" + str(total_samples), end='')
        corners = hmpatch_only_corners(d["hm_x"],d["hm_y"],np.rad2deg(d["G"]),stride,scale=1)
        color_box='#cc0000' #red
        color_box_sk= [1.0,0.0,0.0] #red
        alpha_box=y_pred[i][0]
        if y_pred[i][1]>0.5:
            color_box='#73d216' #geen
            color_box_sk= [0.0,1.0,0.0] #geen
            alpha_box=y_pred[i][1]
        # plot the respective traversability patches on an image
        s_patch = skimage.draw.polygon([ corners[4,1]-stride/2,  corners[4,1]+stride/2, corners[4,1]+stride/2, corners[4,1]-stride/2, corners[4,1]-stride/2], [ corners[4,0]-stride/2, corners[4,0]-stride/2, corners[4,0]+stride/2, corners[4,0]+stride/2, corners[4,0]-stride/2])
        skimage.draw.set_color(sk_hm, (s_patch[0],s_patch[1]), color_box_sk, alpha = alpha_box)
    # for ploting with skimage
    sk_hm_pure = np.copy(sk_hm)
    s_patch = skimage.draw.polygon( arrow_points_sk[[0,1,2,3,0],1], arrow_points_sk[[0,1,2,3,0],0])    
    skimage.draw.set_color(sk_hm, (s_patch[0],s_patch[1]), [0.0,0.0,1.0], alpha = 1.0)
    ax1.imshow(sk_hm)
    
    skimage.io.imsave(heightmap_png[:-4] + '_out_' + ("%.3f" % rad_ori) + '.png' ,sk_hm)

    endTime = time.time()
    workTime =  endTime - startTime
    print ("\rProcessed "+ str(total_samples) + "/" + str(total_samples))
    print ("-- time: "+ str(workTime))    
    #plt.show()
    return sk_hm_pure

   
#%%
# Generating the datasets for training and evaluation/real evaluation
#
height_scale_factor = 1.0
dataset_training, heightmaps_training = generate_composite_dataset_cnn(training_meta_file)
dataset_evaluation, heightmaps_evaluation = generate_composite_dataset_cnn(evaluation_meta_file)

dataset_training.to_csv("dataset_training")                 #saving pandas dataframes for training and evluation
dataset_evaluation.to_csv("dataset_evaluation")

#evaluation_meta_file= "meta_evaluation_real.csv"
#height_scale_factor = 3.0 #for the grave pitmax height is 3m
#dataset_evaluation_real, heightmaps_evaluation_real = generate_composite_dataset_cnn(evaluation_meta_file_real)

#dataset_evaluation_real.to_csv("dataset_evaluation_real")  #saving padnas dataframe for evaluation real
#%%  
# Selecting a subset of the evaluation dataset for evaluation
#
height_scale_factor = 1.0
X_te,y_te=mktrte(dataset_evaluation, heightmaps_evaluation,10000,patch_size)

np.save("X_te",X_te)    # saving evaluation X and y arrays
np.save("y_te",y_te)

# for the real evaluation dataset
#height_scale_factor = 3.0
#X_te_real,y_te_real=mktrte(dataset_evaluation_real, heightmaps_evaluation_real,10000,patch_size)
#height_scale_factor = 1.0

#np.save("X_te_real",X_te)
#np.save("y_te_real",y_te)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# CNN architecture definition + Training
#
#%% 
#  Definition of our neural network architecture
#

model = Sequential()
model.add(Conv2D(5, (3,3), padding='valid', input_shape=(patch_size_training,patch_size_training,1)) )
model.add(Activation('relu'))
model.add(Conv2D(5, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(5, (3,3) ))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer="adadelta",
              metrics=['accuracy'])

#%%
# CNN training
#

batch_size = 150
history=model.fit_generator(
                    generator_tr(dataset_training, heightmaps_training,batch_size,patch_size),
                    steps_per_epoch=100, # usually it is number_of_unique_samples/bach_size
                    epochs=50, 
                    verbose=1,
                    validation_data=(X_te, y_te),
                    #
                    #validation_data=generator_ev(dataset_evaluation, heightmaps_evaluation,batch_size,patch_size),
                    #validation_steps=6,
                    #callbacks=[keras.callbacks.TensorBoard(log_dir='./logs/'+time.strftime("%Y%m%d%H%M%S"), histogram_freq=0, write_graph=False, write_images=False)]
                    callbacks=[custom_callbacks,keras.callbacks.TensorBoard(log_dir='./logs/'+time.strftime("%Y%m%d%H%M%S"), histogram_freq=0, write_graph=False, write_images=False)]
                    )

#
# to visualize the logs on tensorboard:
# >   tensorboard --logdir=dir_where_logs_are_stored

#%% NOTE:
# Once the model is trained is better to save it and use the functions in the 
# script evaluation.py to test the model and build the graph.
# The current script has simmilar funcitons but they are not up to date with
# some perks that evaluation.py has already implemented

model.save(output_folder+"traversability_pioneer_b150_spe_50_e_50_patch_40_acc_XX.h5")

#%% Preliminary testing of the cnn fitted model on other heightmaps

#model.save(output_folder+"traversability_pioneer_b150_spe_100_e_50_acc_82.h5")
model = keras.models.load_model(output_folder+"traversability_pioneer_b150_spe_100_e_50_acc_82.h5")


heightmap_png = heightmaps_folder+ "quarry_cropped4_scaled2cm.png" # quarry heightmap
patch_size = 100 #robot2
patch_resize = 100
stride = 5
sim_hm_mx_x = 15.2 #~30x30m
sim_hm_mx_y = 15.2
height_scale_factor = 10.0


#% for a custom artificial map
heightmap_png = heightmaps_folder+"custom2.png"
patch_size = 60 
patch_resize = 60
stride = 3
sim_hm_mx_x = 5.0
sim_hm_mx_y = 5.0
height_scale_factor = 1.0

#for the multiprocessing option, we need to charge the hm as global becasue some issue with skimage 
#warping treating the fuction argument as a modifiable argument (raed-only source argument error from joblib)
multiprocessing = True
#multiprocessing_hm = skimage.color.rgb2gray(skimage.io.imread(test_height_map_full_png))
#multiprocessing_hm = multiprocessing_hm * height_scale_factor
multiprocessing_hm = read_image(heightmap_png)

# for a single orientation (0 is right, )
test_fitted_model_full_map_stride_cnn(model,heightmap_png,patch_size, patch_resize, 0, stride)
test_fitted_model_full_map_stride_cnn(model,heightmap_png,patch_size, patch_resize, np.pi/2, stride)
test_fitted_model_full_map_stride_cnn(model,heightmap_png,patch_size, patch_resize, np.pi, stride)
test_fitted_model_full_map_stride_cnn(model,heightmap_png,patch_size, patch_resize, 3*np.pi/2, stride)

#%% metric evaluation with the full evaluation dataset
#
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

n_classes = 2

print ('--- cnn classifier ---')
print ('> evaluation dataset')
y_score = model.predict(X_te)
y_score_label = model.predict_classes(X_te)
print ('-')
print('acc:', sklearn.metrics.classification.accuracy_score(y_te[:,1],y_score_label))
print('auc:', sklearn.metrics.roc_auc_score(y_te[:,1],y_score[:,1]))
print(classification_report(y_te[:,1], y_score_label))

print ('> evaluation (realworld) dataset ')
y_score = model.predict(X_te_real)
y_score_label = model.predict_classes(X_te_real)
print ('-')
print('acc:', sklearn.metrics.classification.accuracy_score(y_te_real[:,1],y_score_label))
print('auc:', sklearn.metrics.roc_auc_score(y_te_real[:,1],y_score[:,1]))

print(classification_report(y_te_real[:,1], y_score_label))


#%% in case a very narrow evaluation needs to be done (for a specific heightmap)
# use the cherry picking functions to visualize and evaluate the trained classifier

cp_df, cp_hm, cp_X_te, cp_y_te = cherrypick_single_dataset("evaluation_real74.csv", "gravelpit5.png", 10)

y_score = model.predict(cp_X_te)
y_score_label = model.predict_classes(cp_X_te)

print ('-')
print('acc:', sklearn.metrics.classification.accuracy_score(cp_y_te[:,1],y_score_label))
print('auc:', sklearn.metrics.roc_auc_score(cp_y_te[:,1],y_score[:,1]))


cp_df, cp_hm, cp_X, cp_y = cherrypick_single_dataset("rails2_training_1.csv", "rails2.png", _skips=30)



#%% This main definition is mandatory for parallele computing with joblib   
if __name__ == '__main__':
    main()
    