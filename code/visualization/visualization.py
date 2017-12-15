#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# mayavi libraries for 3D renderings
import os
os.environ["QT_API"] = "pyqt"

import numpy as np
import mayavi.mlab as mlab
import scipy
import skimage
import skimage.transform
import skimage.io
from scipy.misc import imread
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import math

# for arrows
from tvtk.tools import visual

from tqdm import tqdm

import yaml
import operator

import csv

import networkx as nx
import json
#%% utility functions

def readTerrain(name,pixelsize=0.02,max_height=1.0):
    """Returns a tuple terrain,mask with the requested pixelsize
    Values in terrain (2D np float array) are interpreted as meters
    Values in mask (2D np float array) are interpreted as probabilities [0-1]
    """
    
    if(name=="quarry"): 
        tpixelsize=pixelsize
        im=imread("inputs/quarry_cropped4_scaled2cm.png")/2**16*10
        noise=0.0 
        im=im+np.random.rand(*im.shape)*noise
        sz=1.5 # 0: no filtering
        im=scipy.ndimage.filters.gaussian_filter(im, sz, truncate=2) # careful, may create problems with nans
    elif(name=="uzh_o"):
        im=imread("inputs/uzh_elevation.png")[:,:,0].astype(float)/255*5
        im[im==0]=np.nan
        im=skimage.transform.resize(im,np.array(im.shape)*5)
        tpixelsize=0.05
        im=scipy.ndimage.filters.gaussian_filter(im, 3, truncate=1)
    elif(name=="uzh"):
        im=imread("inputs/gridmap_elevation_uzh.png")[:,:,0].astype(float)/255*2.648927
        im[im==0]=np.nan
        im=skimage.transform.resize(im,np.array(im.shape))
        tpixelsize = pixelsize
    elif(name=="elevation_1"):
        im=imread("inputs/gridmap_elevation_1.png")[:,:,0].astype(float)/255*5
        im[im==0]=np.nan
        im=skimage.transform.resize(im,np.array(im.shape))
        tpixelsize = pixelsize
    elif(name=="elevation_2"):
        im=imread("inputs/gridmap_elevation_2_c.png")[:,:,0].astype(float)/255*5
        im[im==0]=np.nan
        im=skimage.transform.resize(im,np.array(im.shape))
        tpixelsize = pixelsize
    elif(name=="elevation_2_r"):
        im=imread("inputs/gridmap_elevation_2_c_r.png")/255*0.739174
        im[im==0]=np.nan
        im=skimage.transform.resize(im,np.array(im.shape))
        tpixelsize = pixelsize
    elif(name=="custom1"):
        im=imread("inputs/custom1.png")[:,:,0].astype(float)/255*1
        tpixelsize=pixelsize      
    elif(name=="custom9"):
        im=imread("inputs/custom9.png")[:,:,0].astype(float)/255*1
        tpixelsize=pixelsize
    elif(name=="arc_rocks"):
        im=imread("inputs/arc_rocks.png").astype(float)/255*0.4
        tpixelsize=pixelsize
    elif(name=="heightmap1"):
        im=imread("inputs/heightmap1.png")[:,:,0].astype(float)/255*1
        tpixelsize=pixelsize
    else:
        im=imread("inputs/"+name+".png").astype(float)/255*max_height
        tpixelsize=pixelsize        
        #assert(False)
    im=skimage.transform.resize(im,(np.array(im.shape)*tpixelsize/pixelsize).astype(int))
    im=im-np.nanmin(im)
    
    return im


def makeRobot(x,y,z,length,angle=0,fontsize=0.4):
    vertices=np.array([[0,0,1],[1,0,0],[0,3,0],[-1,0,0]]).astype(float)/3*length+np.array([x,y,z])
    triangles=[[0,1,3],[0,1,2],[0,3,2],[3,1,2]]
    mesh=mlab.triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], triangles, color=(0.3,0.3,0.6))
    mesh.actor.property.interpolation = 'phong'
    mesh.actor.property.specular = 0.2
    mesh.actor.property.specular_power = 10
    mesh.actor.property.ambient_color = (1,1,1)
    mesh.actor.property.diffuse_color = (0.3,0.3,0.6)
    mesh.actor.property.color = (0.3,0.3,0.6)
    mesh.actor.property.ambient = 0.1
    mesh.actor.actor.rotate_z(angle)
    text = mlab.text3d(vertices[2,0]*math.cos(math.radians(angle))-vertices[2,1]*math.sin(math.radians(angle)), vertices[2,1]*math.cos(math.radians(angle))+vertices[2,0]*math.sin(math.radians(angle)), vertices[2,2], "robot", scale=fontsize, color=(0,0,0))
    return()

def makeRobot2(x,y,z,length,angle=0,fontsize=0.4):
    # with our own transformation implementation
    vertices=np.array([[0,0,1],[0,1,0],[-3,0,0],[0,-1,0]]).astype(float)
    verticesp=np.copy(vertices)
    verticesp[:,0]=vertices[:,0]*math.cos(math.radians(angle))-vertices[:,1]*math.sin(math.radians(angle))
    verticesp[:,1]=vertices[:,1]*math.cos(math.radians(angle))+vertices[:,0]*math.sin(math.radians(angle))
    vertices = verticesp/3*length+np.array([x,y,z])
    triangles=[[0,1,3],[0,1,2],[0,3,2],[3,1,2]]
    
    # plot a kind of arrow to represent the robot when stl model is not present
    '''
    mesh=mlab.triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], triangles, color=(0.3,0.3,0.6))
    mesh.actor.property.interpolation = 'phong'
    mesh.actor.property.specular = 0.2
    mesh.actor.property.specular_power = 10
    mesh.actor.property.ambient_color = (1,1,1)
    mesh.actor.property.diffuse_color = (0.3,0.3,0.6)
    mesh.actor.property.color = (0.3,0.3,0.6)
    mesh.actor.property.ambient = 0.1
    '''
    text = mlab.text3d(vertices[2,0]+0.1, vertices[2,1]+0.1, vertices[2,2], "robot", scale=fontsize, color=(0,0,0))
    
    # plot the stl file of the pioneer p3at robot
    STLfile="inputs/pioneer3at.stl"
    f=open(STLfile,'r')
    
    x=[]
    y=[]
    z=[]
    
    for line in f:
    	strarray=line.split()
    	if strarray[0]=='vertex':
    		x=np.append(x,np.double(strarray[1]))
    		y=np.append(y,np.double(strarray[2]))
    		z=np.append(z,np.double(strarray[3]))
    
    triangles=[(i, i+1, i+2) for i in range(0, len(x),3)]
    
    mlab.triangular_mesh(vertices[0,0]+  x*math.cos(math.radians(angle))-y*math.sin(math.radians(angle))  , vertices[0,1]+ y*math.cos(math.radians(angle))+x*math.sin(math.radians(angle)) , vertices[0,2]+z, triangles, color=(0.2,0.2,1.0))
    mlab.show()
    
    return()


def visualizeTerrain(terrain, pixelsize, mask=None, fig=None, fontsize=0.4, robot_pose=(-1,0,0,0.8,0), details=3):
    # Original version: not very flexible to display traversability mask.
    # Use visualizeTerrain2 instead if you have a mask
    if(fig is None):
        fig = mlab.figure()
    
    fig.scene.background = (1,1,1)
    
    y, x = np.meshgrid(np.arange(terrain.shape[0])*pixelsize,np.arange(terrain.shape[1])*pixelsize)
    s = mlab.surf(x, y, terrain, color=(1.0,1.0,1.0))
    
    s.actor.property.interpolation = 'phong'
    s.actor.property.specular = 0.0
    s.actor.property.specular_power = 10
    s.actor.property.ambient_color = (1,1,1)
    s.actor.property.diffuse_color = (0.7,0.7,0.7)
    s.actor.property.color = (0.7,0.7,0.7)
    s.actor.property.ambient = 0.02
    
    if(mask is not None):
        terrain_mask=terrain.copy()
        terrain_mask[mask<0.9]=np.nan
        sm = mlab.surf(x, y, terrain_mask+0.010, color=(1.0,1.0,1.0))
    
        sm.actor.property.interpolation = 'phong'
        sm.actor.property.specular = 0.2
        sm.actor.property.specular_power = 10
        sm.actor.property.ambient_color = (0.6,1,0.6)
        sm.actor.property.diffuse_color = (0.6,0.9,0.6)
        sm.actor.property.color = (0.6,0.9,0.6)
        sm.actor.property.opacity = 0.7
        sm.actor.property.ambient = 0.05
        
        terrain_mask=terrain.copy()
        terrain_mask[mask<0.2]=np.nan
        sm = mlab.surf(x, y, terrain_mask+0.005, color=(1.0,1.0,1.0))
    
        sm.actor.property.interpolation = 'phong'
        sm.actor.property.specular = 0.2
        sm.actor.property.specular_power = 10
        sm.actor.property.ambient_color = (0.9,0.9,0.6)
        sm.actor.property.diffuse_color = (0.9,0.9,0.6)
        sm.actor.property.color = (0.8,0.8,0.6)
        sm.actor.property.opacity = 0.5
        sm.actor.property.ambient = 0.05
    
    if details > 0:
        square=np.array([[0,0],[1,0],[1,1],[0,1]])[[0,1,2,3,0],:]
        square=np.hstack((square*np.array([[np.max(y),np.max(x)]]), np.zeros((5,1))))
        base=mlab.plot3d(square[:,0], square[:,1], square[:,2], color=(0,0,0), line_width=2)
    if details > 1:
        for i in range(4):
            if i == 1 or i == 2:
                p=np.mean(square[[i,i+1],:],axis=0)
                d=np.linalg.norm(square[i+1,:]-square[i+0,:])
                mlab.text3d(p[0]+1.1, p[1]+0.4, p[2], "{:.1f}m".format(d), scale=fontsize, color=(0,0,0))
        height=mlab.plot3d(np.array([0.0,0.0]), np.array([0.0,0.0]), np.array([0.0,np.nanmax(terrain)]), color=(0,0,0), line_width=2)
        #mlab.text3d(0.0, 0.0, np.nanmax(terrain)/2, "{:.1f}m".format(np.nanmax(terrain)), scale=fontsize, color=(0,0,0))
        mlab.text3d(0.0, 0.0, np.nanmax(terrain)*0.9, "{:.1f}m".format(np.nanmax(terrain)), scale=fontsize, color=(0,0,0))
    if details > 2:
        makeRobot2(robot_pose[0],robot_pose[1],robot_pose[2],robot_pose[3],angle=robot_pose[4], fontsize=fontsize)    
    
    from tvtk.api import tvtk
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

def visualizeTerrainPatch(terrain, pixelsize, mask=None, fig=None, fontsize=0.4, robot_pose=(-1,0,0,0.8,0), stext = True, border_color=(0,0,0)):
    # visualize sample patches of a heightmap
    if(fig is None):
        fig = mlab.figure()
    
    fig.scene.background = (1,1,1)
    
    y, x = np.meshgrid(np.arange(terrain.shape[0])*pixelsize,np.arange(terrain.shape[1])*pixelsize)
    s = mlab.surf(x, y, terrain, color=(1.0,1.0,1.0))
    
    s.actor.property.interpolation = 'phong'
    s.actor.property.specular = 0.0
    s.actor.property.specular_power = 10
    s.actor.property.ambient_color = (1,1,1)
    s.actor.property.diffuse_color = (0.7,0.7,0.7)
    s.actor.property.color = (0.7,0.7,0.7)
    s.actor.property.ambient = 0.02
    
    if(mask is not None):
        terrain_mask=terrain.copy()
        terrain_mask[mask<0.9]=np.nan
        sm = mlab.surf(x, y, terrain_mask+0.010, color=(1.0,1.0,1.0))
    
        sm.actor.property.interpolation = 'phong'
        sm.actor.property.specular = 0.2
        sm.actor.property.specular_power = 10
        sm.actor.property.ambient_color = (0.6,1,0.6)
        sm.actor.property.diffuse_color = (0.6,0.9,0.6)
        sm.actor.property.color = (0.6,0.9,0.6)
        sm.actor.property.opacity = 0.7
        sm.actor.property.ambient = 0.05
        
        terrain_mask=terrain.copy()
        terrain_mask[mask<0.2]=np.nan
        sm = mlab.surf(x, y, terrain_mask+0.005, color=(1.0,1.0,1.0))
    
        sm.actor.property.interpolation = 'phong'
        sm.actor.property.specular = 0.2
        sm.actor.property.specular_power = 10
        sm.actor.property.ambient_color = (0.9,0.9,0.6)
        sm.actor.property.diffuse_color = (0.9,0.9,0.6)
        sm.actor.property.color = (0.8,0.8,0.6)
        sm.actor.property.opacity = 0.5
        sm.actor.property.ambient = 0.05
    
    square=np.array([[0,0],[1,0],[1,1],[0,1]])[[0,1,2,3,0],:]
    square=np.hstack((square*np.array([[np.max(y),np.max(x)]]), np.zeros((5,1))))
    base=mlab.plot3d(square[:,0], square[:,1], square[:,2], color=border_color, line_width=2)
    for i in range(4):
        p=np.mean(square[[i,i+1],:],axis=0)
        d=np.linalg.norm(square[i+1,:]-square[i+0,:])
        if stext:
            mlab.text3d(p[0], p[1], p[2], "{:.1f}m".format(d), scale=fontsize, color=(0,0,0))
    height=mlab.plot3d(np.array([0.0,0.0]), np.array([0.0,0.0]), np.array([0.0,np.nanmax(terrain)]), color=(0,0,0), line_width=2)
    #mlab.text3d(0.0, 0.0, np.nanmax(terrain)/2, "{:.1f}m".format(np.nanmax(terrain)), scale=fontsize, color=(0,0,0))
    if stext:
        mlab.text3d(0.0, 0.0, np.nanmax(terrain)*0.9, "{:.1f}m".format(np.nanmax(terrain)), scale=fontsize, color=(0,0,0))
    
    if stext == False:
        fontsize = 0.0
    if robot_pose != (-1000,-1000,-1000,-1000,-1000):
        makeRobot2(robot_pose[0],robot_pose[1],robot_pose[2],robot_pose[3],angle=robot_pose[4], fontsize=fontsize)    
    
    # grid
    color_grid = (42/255,56/255,54/255)
    for xi in [int(np.floor(k)) for k in np.linspace(0,x.shape[1]-1,5)]:
        mlab.plot3d(x[:,xi],y[:,xi],terrain[:,xi]+0.002, color=color_grid, line_width=1, tube_radius=None)
    for yi in [int(np.floor(k)) for k in np.linspace(0,x.shape[0]-1,5)]:
        mlab.plot3d(x[yi,:],y[yi,:],terrain[yi,:]+0.002, color=color_grid, line_width=1, tube_radius=None)
       
    from tvtk.api import tvtk
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()


def visualizeTerrain2(terrain, pixelsize, mask, fig=None, fontsize=0.4, robot_pose=(-1,0,0,0.8,0)):
    # Draws a surface colored according to mask using a custom colormap
    # Inspired by http://gael-varoquaux.info/programming/mayavi-representing-an-additional-scalar-on-surfaces.html
    
    if(fig is None):
        fig = mlab.figure()
    
    fig.scene.background = (1,1,1)
    
    y, x = np.meshgrid(np.arange(terrain.shape[0])*pixelsize,np.arange(terrain.shape[1])*pixelsize)
    s = mlab.mesh(x, y, terrain, scalars=mask) # We use a mesh, not a surf. Note: it's less efficient
    
    # inspired by http://docs.enthought.com/mayavi/mayavi/auto/example_custom_colormap.html
    #colormap=np.tile(np.array([255, 102, 102,255]),[256,1]) # all red
    colormap=np.tile(np.array([180,180,180,255]),[256,1]) # all gray
    colormap[:,1]=np.linspace(180,250,256) # scale green channel

    # 0 scalar value will be gray (180,180,180), the rest is from yellow to green
    #colormap[:,0]=np.concatenate(([255],np.linspace(220,100,255))) # swipe red channel downwards to go from yellow to green
    #colormap[:,1]=np.concatenate(([255],np.ones((255))*220)) # a tone green stays always up
    #colormap[:,2]=np.concatenate(([255],np.ones((255))*100)) # blue is al 0 excepto from 0 where the gray will be
    s.module_manager.scalar_lut_manager.lut.table = colormap
    s.module_manager.scalar_lut_manager.lut.range=np.array([0.0,1.0])
    s.actor.property.interpolation = 'phong'
    s.actor.property.specular = 0.0
    s.actor.property.specular_power = 10
    s.actor.property.ambient_color = (1,1,1)

    s.actor.property.ambient = 0.02
    
    square=np.array([[0,0],[1,0],[1,1],[0,1]])[[0,1,2,3,0],:]
    square=np.hstack((square*np.array([[np.max(x),np.max(y)]]), np.zeros((5,1))))
    base=mlab.plot3d(square[:,0], square[:,1], square[:,2], color=(0,0,0), line_width=2)
    for i in range(4):
        p=np.mean(square[[i,i+1],:],axis=0)
        d=np.linalg.norm(square[i+1,:]-square[i+0,:])
        mlab.text3d(p[0], p[1], p[2], "{:.1f}m".format(d), scale=fontsize, color=(0,0,0))
    height=mlab.plot3d(np.array([0.0,0.0]), np.array([0.0,0.0]), np.array([0.0,np.nanmax(terrain)]), color=(0,0,0), line_width=2)
    mlab.text3d(0.0, 0.0, np.nanmax(terrain)*1.0, "{:.1f}m".format(np.nanmax(terrain)), scale=fontsize, color=(0,0,0))
    
    makeRobot2(robot_pose[0],robot_pose[1],robot_pose[2],robot_pose[3], angle=robot_pose[4], fontsize=fontsize)
    
    from tvtk.api import tvtk
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    
def visualizePatch(patch, pixelsize, fig=None, robotlength=0.8):
    # simplified version of visualizeTerrainPatch
    if(fig is None):
        fig = mlab.figure()
    fig.scene.background = (1,1,1)
    
    patch=patch-np.nanmin(patch)
    y, x = np.meshgrid(np.arange(patch.shape[0])*pixelsize,np.arange(patch.shape[1])*pixelsize)
    s = mlab.surf(x, y, patch, color=(1.0,1.0,1.0))
    s.actor.property.interpolation = 'phong'
    s.actor.property.specular = 0.0
    s.actor.property.specular_power = 10
    s.actor.property.ambient_color = (1,1,1)
    s.actor.property.diffuse_color = (0.7,0.7,0.7)
    s.actor.property.color = (0.7,0.7,0.7)
    s.actor.property.ambient = 0.02
        
    square=np.array([[0,0],[1,0],[1,1],[0,1]])[[0,1,2,3,0],:]
    square=np.hstack((square*np.array([[np.max(x),np.max(y)]]), np.zeros((5,1))))
    base=mlab.plot3d(square[:,0], square[:,1], square[:,2], color=(0,0,0), line_width=2, tube_radius=None)
    for i in range(4):
        p=np.mean(square[[i,i+1],:],axis=0)
        d=np.linalg.norm(square[i+1,:]-square[i+0,:])
    
    for xi in [int(np.floor(k)) for k in np.linspace(0,x.shape[1]-1,5)]:
        mlab.plot3d(x[:,xi],y[:,xi],patch[:,xi]+0.002, color=(0,0,0), line_width=1, tube_radius=None)
    for yi in [int(np.floor(k)) for k in np.linspace(0,x.shape[0]-1,5)]:
        mlab.plot3d(x[yi,:],y[yi,:],patch[yi,:]+0.002, color=(0,0,0), line_width=1, tube_radius=None)
    
    makeRobot(np.mean(x),-robotlength*1.2,0,length=robotlength)
    
    mlab.view(azimuth=-90, elevation=45)
    
    from tvtk.api import tvtk
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()


def visualizePathsonTerrain(terrain, paths_file_yaml, offset = (30,30), pixelsize = 0.02, stride = 5, robot_pose=(-1,0,0,0.8,0), fontsize=0.4, out_name="", all_paths=True):
    # Reads a path file (yaml) and plots the set of paths in such file 
    # on the respective heightmap rendering
    # paths files are generated by the pareto path planner
    fig=mlab.figure(size=(1000,1000))
    visualizeTerrain(terrain,pixelsize,fig=fig,fontsize=fontsize,robot_pose=robot_pose)
    # for each graph, we use a specific stride, use that for conversion
    with open(paths_file_yaml, 'r') as stream:
        paths = yaml.load(stream)
    
    o_paths = [(i,e['prob']) for i,e in enumerate(paths)]
    o_paths = sorted(o_paths, key = lambda e:e[1])
    
    #plot all the paths if true
    if all_paths == False:
        #select a patch for each slice (0.1) of probability
        th = 0
        t_paths = []
        for e in o_paths:
            if e[1] >= th:
                t_paths.append(e)
                th = th + 0.1
                if th < e[1]:
                    th = float(str(e[1])[:3]) + 0.1
        o_paths = t_paths

    # where the sampling started when building the traversability graph, usally is (size_patch/2, size_patch/2)
    off_x = offset[0] * pixelsize
    off_y = offset[1] * pixelsize
    print ("source_p: ", paths[0]['path'][0]['i'], ', ', paths[0]['path'][0]['j'])    
    for i_path in o_paths:
        path = paths[i_path[0]]
        print ("distance: ", path['dist'], " probability: ", path['prob'], " path legth: ", np.shape(path['path']))
        waypoints = []
        for segment in path['path']:
            #id, dist, i, j
            if False: # when elements i and j are correct, just use them
                waypoints.append([off_x+int(segment['i'])*stride*pixelsize,off_y+int(segment['j'])*stride*pixelsize])   
            else: # calculate them using the imsize, stride, offset and node id
                nidx = int((terrain.shape[0]-2*offset[0])/stride)
                nidy = int((terrain.shape[1]-2*offset[1])/stride)
                ni= int(segment['id'])/nidx
                nj= int(segment['id'])%nidy
                waypoints.append([off_x+int(ni)*stride*pixelsize,off_y+int(nj)*stride*pixelsize])
        waypoints=np.array(waypoints)
        if True : #smoothing
            # rough smoothing, taking the the middle point of each segment in waypoints
            # use a finest smoothing afterwards
            s_waypoints = [ ( (waypoints[i,0]+waypoints[i+1,0])/2 , (waypoints[i,1]+waypoints[i+1,1])/2 ) for i in range(0,waypoints.shape[0]-1)]
            waypoints = np.concatenate((waypoints[0:1],np.array(s_waypoints),waypoints[-1:,]))
        # function returning height of an xy point (in meters) sampled on im
        y, x = np.meshgrid(np.arange(terrain.shape[0])*pixelsize,np.arange(terrain.shape[1])*pixelsize)
        if False:
            heightof=scipy.interpolate.RectBivariateSpline(x[:, 0], y[0, :], terrain)
        else: #when too many NaN regions in map
            heightof=scipy.interpolate.RectBivariateSpline(x[:, 0], y[0, :], np.nan_to_num(terrain))
        # coordinates in meters on map of my waypoints
        ls=np.cumsum([0]+[np.linalg.norm(end-start) for start,end in zip(waypoints[0:-1],waypoints[1:])])
        psxy=scipy.interpolate.interp1d(ls,waypoints,axis=0)(np.arange(0,ls[-1], 0.1)) # sample one point every 10 cm
        # ps are coordinates of dense points sampled along the path defined by waypoints
        psz=heightof(psxy[:,0],psxy[:,1],grid=False)
        
        mlab.plot3d(psxy[:,0], psxy[:,1], psz + 0.05, color=(1.0-float(path['prob']),float(path['prob']),0.0), opacity = 0.6, tube_radius=0.04)
    nmf_path = "partial"
    if all_paths:
        nmf_path = "all"
    mlab.savefig("paths_"+str(nmf_path)+"_traver_dist_"+out_name+".png",figure=fig)
    return fig


def visualizeOnlyPathsonTerrain(e_fig, terrain, paths_file_yaml, offset = (30,30), pixelsize = 0.02, stride = 5, robot_pose=(-1,0,0,0.8,0), fontsize=0.4, out_name="", list_paths=[]):
    fig=e_fig
    # similar to visualizePathsonTerrain but the heightmap surface is not rendered
    # it assumes that the surfaces has been renderd before
    # only paths + robot are drawn on the e_fig window
    
    makeRobot2(robot_pose[0],robot_pose[1],robot_pose[2],robot_pose[3],angle=robot_pose[4], fontsize=fontsize)
    
    paths_radius = 0.09
    
    with open(paths_file_yaml, 'r') as stream:
        paths = yaml.load(stream)
    
    o_paths = [i for i,e in enumerate(paths)]
    
    # plot all the paths if true
    if list_paths != []:
        o_paths = list_paths

    #where the sampling started when building the graph, usally is (size_patch/2, size_patch/2)
    off_x = offset[0] * pixelsize
    off_y = offset[1] * pixelsize
    print ("source_p: ", paths[0]['path'][0]['i'], ', ', paths[0]['path'][0]['j'])    
    #for path in paths:
    for i_path in o_paths:
        path = paths[i_path]
        print ("distance: ", path['dist'], " probability: ", path['prob'], " path legth: ", np.shape(path['path']))
        waypoints = []
        for segment in path['path']:
            #id, dist, i, j
            if False: # when elements i and j are correct, just use them
                waypoints.append([off_x+int(segment['i'])*stride*pixelsize,off_y+int(segment['j'])*stride*pixelsize])   
            else: # calculate them using the imsize, stride, offset and node id
                nidx = int((terrain.shape[0]-2*offset[0])/stride)
                nidy = int((terrain.shape[1]-2*offset[1])/stride)
                ni= int(segment['id'])/nidx
                nj= int(segment['id'])%nidy
                waypoints.append([off_x+int(ni)*stride*pixelsize,off_y+int(nj)*stride*pixelsize])
        waypoints=np.array(waypoints)
        if True : #smoothing
            # rough smoothing, taking the the middle point of each segment in waypoints
            # use a finest smoothing afterwards
            s_waypoints = [ ( (waypoints[i,0]+waypoints[i+1,0])/2 , (waypoints[i,1]+waypoints[i+1,1])/2 ) for i in range(0,waypoints.shape[0]-1)]
            waypoints = np.concatenate((waypoints[0:1],np.array(s_waypoints),waypoints[-1:,]))
        
        # function returning height of an xy point (in meters) sampled on im
        y, x = np.meshgrid(np.arange(terrain.shape[0])*pixelsize,np.arange(terrain.shape[1])*pixelsize)
        if False:
            heightof=scipy.interpolate.RectBivariateSpline(x[:, 0], y[0, :], terrain)
        else: #when too many NaN regions in map
            heightof=scipy.interpolate.RectBivariateSpline(x[:, 0], y[0, :], np.nan_to_num(terrain))
        # coordinates in meters on map of my waypoints
        ls=np.cumsum([0]+[np.linalg.norm(end-start) for start,end in zip(waypoints[0:-1],waypoints[1:])])
        psxy=scipy.interpolate.interp1d(ls,waypoints,axis=0)(np.arange(0,ls[-1], 0.1)) # sample one point every 10 cm
        # ps are coordinates of dense points sampled along the path defined by waypoints
        psz=heightof(psxy[:,0],psxy[:,1],grid=False)
        
        mlab.plot3d(psxy[:,0], psxy[:,1], psz + 0.05, color=(1.0-float(path['prob']),float(path['prob']),0.0), opacity = 0.7, tube_radius=paths_radius)
    nmf_path = "selection_paths"
    if list_paths == []:
        nmf_path = "all"
    mlab.savefig("paths_"+str(nmf_path)+"_traver_dist_"+out_name+".png",figure=fig)


def visualizeGenerateVrepPathonTerrain(terrain, paths_file_yaml, offset = (30,30), pixelsize = 0.02, stride = 5, robot_pose=(-1,0,0,0.8,0), fontsize=0.4, out_name="", all_paths=True):
    # similar to visualizePathsonTerrain but this one generate a cvs file per 
    # path to be imported in other software (e.g. vrep simulator) and validate paths
    fig=mlab.figure(size=(1000,1000))
    visualizeTerrain(terrain,pixelsize,fig=fig,fontsize=fontsize,robot_pose=robot_pose)
    # for each graph, we use a specific stride, use that for conversion
    with open(paths_file_yaml, 'r') as stream:
        paths = yaml.load(stream)
    
    o_paths = [(i,e['prob']) for i,e in enumerate(paths)]
    o_paths = sorted(o_paths, key = lambda e:e[1])
    
    #plot all the paths if true
    if all_paths == False:
        # select a patch for each slice (0.1) of probability
        th = 0
        t_paths = []
        for e in o_paths:
            if e[1] >= th:
                t_paths.append(e)
                th = th + 0.1
                if th < e[1]:
                    th = float(str(e[1])[:3]) + 0.1
        o_paths = t_paths
    # where the sampling started when building the graph, usally is (size_patch/2, size_patch/2)
    off_x = offset[0] * pixelsize
    off_y = offset[1] * pixelsize
    print ("source_p: ", paths[0]['path'][0]['i'], ', ', paths[0]['path'][0]['j'])    
    nm_p = 0
    for i_path in o_paths:
        path = paths[i_path[0]]
        print ("distance: ", path['dist'], " probability: ", path['prob'], " path legth: ", np.shape(path['path']))
        waypoints = []
        for segment in path['path']:
            #id, dist, i, j
            if False: # when elements i and j are correct, just use them
                waypoints.append([off_x+int(segment['i'])*stride*pixelsize,off_y+int(segment['j'])*stride*pixelsize])   
            else: #calculate them using the imsize, stride, offset and node id
                nidx = int((terrain.shape[0]-2*offset[0])/stride)
                nidy = int((terrain.shape[1]-2*offset[1])/stride)
                ni= int(segment['id'])/nidx
                nj= int(segment['id'])%nidy
                waypoints.append([off_x+int(ni)*stride*pixelsize,off_y+int(nj)*stride*pixelsize])
        waypoints=np.array(waypoints)
        if True : #smoothing
            # rough smoothing, taking the the middle point of each segment in waypoints
            # use a finest smoothing afterwards
            s_waypoints = [ ( (waypoints[i,0]+waypoints[i+1,0])/2 , (waypoints[i,1]+waypoints[i+1,1])/2 ) for i in range(0,waypoints.shape[0]-1)]
            waypoints = np.concatenate((waypoints[0:1],np.array(s_waypoints),waypoints[-1:,]))
        
        # function returning height of an xy point (in meters) sampled on im
        y, x = np.meshgrid(np.arange(terrain.shape[0])*pixelsize,np.arange(terrain.shape[1])*pixelsize)
        if False:
            heightof=scipy.interpolate.RectBivariateSpline(x[:, 0], y[0, :], terrain)
        else: #when too many NaN regions in map
            heightof=scipy.interpolate.RectBivariateSpline(x[:, 0], y[0, :], np.nan_to_num(terrain))
        # coordinates in meters on map of my waypoints
        ls=np.cumsum([0]+[np.linalg.norm(end-start) for start,end in zip(waypoints[0:-1],waypoints[1:])])
        psxy=scipy.interpolate.interp1d(ls,waypoints,axis=0)(np.arange(0,ls[-1], 0.1)) # sample one point every 10 cm
        # ps are coordinates of dense points sampled along the path defined by waypoints
        psz=heightof(psxy[:,0],psxy[:,1],grid=False)
        
        print ("----- ",psxy.shape)
        
        path_csv = []
        for pi in range(0,psz.shape[0]-1):
            row = dict()
            # remember to convert to VREP reference, we are on still on MAYAVI
            # this depends on the heightmap size
            h_mx = 15.2
            h_my = 15.2
            row['x'] = psxy[pi,1] - h_mx
            row['y'] = -1.0*(psxy[pi,0] - h_my)
            row['z'] = psz[pi]+0.15
            # these can also be ommited
            row['alpha'] = 0
            row['beta'] = 0
            t_g = math.atan2( (-1.0*(psxy[pi+1,0] - h_my)) - float(row['y']) , (psxy[pi+1,1] - h_mx) - float(row['x']))
            if t_g <= math.pi/2:
                row['gamma'] = 2*math.pi - (math.pi/2 - t_g)
            else:
                row['gamma'] = t_g - math.pi/2
            
            # rest of values can be ommited (distance, flags, ...)
            #
            path_csv.append(row)
        with open("paths_traver_dist_"+out_name+"_"+str(nm_p)+".csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in path_csv:
                spamwriter.writerow([row['x'], row['y'], row['z'],row['alpha'],row['beta'],row['gamma']])
        
        nm_p = nm_p + 1
        mlab.plot3d(psxy[:,0], psxy[:,1], psz + 0.05, color=(1.0-float(path['prob']),float(path['prob']),0.0), opacity = 0.6, tube_radius=0.04)
    nmf_path = "partial"
    if all_paths:
        nmf_path = "all"
    mlab.savefig("paths_"+str(nmf_path)+"_traver_dist_"+out_name+".png",figure=fig)



def visualizeReachabilityonTerrain(terrain, reachability_file_yaml, offset = (30,30), pixelsize = 0.02, stride = 5, robot_pose=(-1,0,0,0.8,0), fontsize=0.4, out_name=""):
    # another traversability represantion is reachability
    # from a robot-s position, how travesable is the rest of the heightmap
    # using the trained model?
    #
    # reachability files (yaml) are also generated by the pareto planner
    #
    with open(reachability_file_yaml, 'r') as stream:
        reachability = yaml.load(stream)
    
    # here we save the reachability value from yaml
    mask = np.zeros((terrain.shape[0],terrain.shape[1]),dtype='float64') 
    
    for idx, prob in reachability.items():
        # transform idx in i, j coordinates
        nidx = int((terrain.shape[0]-2*offset[0])/stride)
        nidy = int((terrain.shape[1]-2*offset[1])/stride)
        ni= int(idx)/nidx
        nj= int(idx)%nidy
        # then to image coordinates (center of the patch)
        imx = offset[0] + ni*stride
        imy = offset[1] + nj*stride
        # draw a square with the coresponding proba value        
        r = np.array([imx-stride/2, imx+stride/2, imx+stride/2, imx-stride/2, imx-stride/2])
        c = np.array([imy-stride/2, imy-stride/2, imy+stride/2, imy+stride/2, imy-stride/2])
        rr, cc = skimage.draw.polygon(r, c, terrain.shape)
        mask[rr, cc] = prob
    
    # plot the map and the generated reachability mask
    fig=mlab.figure(size=(1000,1000))
    visualizeTerrain2(terrain,pixelsize,mask=mask, fig=fig, fontsize=0.2, robot_pose=robot_pose )
    mlab.savefig("reachability"+out_name+".png",figure=fig)
    #mlab.close(fig)

def saveAnimation(fig, directory, motion="360"):
    # Saves an animation from the passed mlib figure in the provided directory
    # Will create directory if it does not exist, remove any previous frames in 
    # the directory if any exists.
    # Individual frames are saved as PNG, motion can be ["360", "oscillate"]
    # Motion begins at initial view in figure
    
    import pathlib
    directory=pathlib.Path(directory) # Converts in case it's a string
    directory.mkdir(exist_ok=True)
    assert(directory.is_dir())
    [f.unlink() for f in directory.glob("anim*.png")] # remove any preexisting animation in the directory
    
    (original_azimuth, elevation, distance, focalpoint)=mlab.view(figure=fig)
    
    azimuths={"360":        np.arange(0,360,2)+original_azimuth,
              "oscillate":  np.sin(np.linspace(0,2*np.pi,100,endpoint=False))*30+original_azimuth}[motion]
    for i,a in tqdm(list(enumerate(azimuths))):
        mlab.view(azimuth=a,figure=fig)
        mlab.savefig(str(directory/'anim{:04d}.png'.format(i)))
    
    import subprocess
    subprocess.call(["convert", "-delay", "1x20", "-loop", "0", str(directory)+"/anim*.png", str(directory)+"/anim.mp4"])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Generation of 3D renderings for heightmaps, oriented traversability maps, 
# minimal traversability maps, paths, reachability maps, and animations
################################################################################# 


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# reading the quarry heightmap and rendering it
#
pixelsize=0.02
terrain=readTerrain("quarry",pixelsize)
mlab.close(all=True)
mlab.options.offscreen = True # To avoid opening a window
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,fig=fig)
#
# optionally, an animation can be created if needed
#
#saveAnimation(fig, "terrain-360", motion="360")
#saveAnimation(fig, "terrain-osc", motion="oscillate")
# remember to comment next line to avoid open and closing the rendering
# mlab.close(fig)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#
# visualize different orientation overlays for a heightmap
#

terrain_map = readTerrain("quarry",pixelsize=pixelsize)
t_overlays = skimage.io.imread_collection("inputs/t_orientations/quarry_cropped4_scaled2cm/*.png");
i=1
o_step = 360/np.shape(t_overlays)[0]
mlab.close(all=True)
for terrain in t_overlays:
    # traversability is in alpha channel
    terrain=terrain[:,:,3].astype(float)/255*1.0
    fig=mlab.figure(size=(1000,1000))
    visualizeTerrain2(terrain_map,pixelsize,mask=terrain, fig=fig, robot_pose=(31,33,0,0.8,(i-1)*o_step - 90))# ,fontsize=0.1,robot_pose=(1.5,-1.0,0,0.8,90), stext=False)
    # arrow:
    visual.set_viewer(fig)
    angle = (i-1)*o_step
    # as we simulate pioneer on gazebo the rorientaiton frame is differentfrom the first robot, 0 is -> right, counter-clockwise
    xp=(-0)*math.cos(math.radians(angle))-(1)*math.sin(math.radians(angle))
    yp=(1)*math.cos(math.radians(angle))+(-0)*math.sin(math.radians(angle))
    arrow=visual.Arrow(x=16,y=39,z=0, color=(0.2,0.2,0.8), axis=(xp,yp,0), length_cone=0.3, radius_cone =0.17, radius_shaft=0.08)
    arrow.actor.scale=[4,3,1]
    arrow.pos = arrow.pos/[4,3,1]
    arrow.actor.property.interpolation = 'phong'
    arrow.actor.property.specular = 0.2
    arrow.actor.property.specular_power = 10
    arrow.actor.property.ambient_color = (1,1,1)
    arrow.actor.property.diffuse_color = (0.3,0.3,0.7)
    arrow.actor.property.color = (0.2,0.2,0.8)
    arrow.actor.property.ambient = 0.1
    #
    # creates a 3D rendering for each oriented map and save as an image
    mlab.savefig("t_orientation_"+str((i-1)*o_step)+".png",figure=fig)
    #saveAnimation(fig, "terrain-360", motion="360")
    #saveAnimation(fig, "terrain-osc", motion="oscillate")
    mlab.close(fig)
    i = i+ 1
#
# this is the mininmal traversability map 
#
mask=imread("inputs/min_traversability_map_color_quarry.png")[:,:,3].astype(float)/255
fig=mlab.figure(size=(1000,1000))
visualizeTerrain2(terrain_map,pixelsize,mask=mask, fig=fig, fontsize=0.4, robot_pose=(31,33,0,0.8,0) )
#
# creates an animation of the provided minimal traversability map
# that is currently displayed in the rendering window
#
saveAnimation(fig, "terrain-360", motion="360")
mlab.savefig("min_t_map_quarry.png",figure=fig)
#mlab.close(fig)


#%%
#
# Examples for other heightmaps
#
pixelsize=0.1
terrain=readTerrain("uzh",pixelsize)
mlab.close(all=True)
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,fig=fig)
# remember to comment next line to avoid open and closing the rendering
# mlab.close(fig)

#%%
pixelsize=0.03
terrain=readTerrain("elevation_1",pixelsize)
mlab.close(all=True)
#mlab.options.offscreen = True # To avoid opening a window
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,fig=fig)
# remember to comment next line to avoid open and closing the rendering
# mlab.close(fig)

#%%
pixelsize=0.03
terrain=readTerrain("elevation_2",pixelsize)
mlab.close(all=True)
#mlab.options.offscreen = True # To avoid opening a window
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,fig=fig)
# remember to comment next line to avoid open and closing the rendering
# mlab.close(fig)

#%%
#
# To only show the minimal traversability map on top of a rendered heightmmap
#
pixelsize=0.02
terrain=readTerrain(name="quarry",pixelsize=pixelsize)
mask=imread("inputs/min_traversability_map_color_quarry.png")[:,:,3].astype(float)/255
#mlab.options.offscreen = False # To avoid opening a window
fig=mlab.figure(size=(1000,1000))
#visualizeTerrain(terrain,pixelsize,mask=mask,fig=fig)
visualizeTerrain2(terrain,pixelsize,mask=mask,fig=fig)
# remember to comment next line to avoid open and closing the rendering
# mlab.close(fig)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#
# Visualize some of the heightmaps from the training and evaluation dataset
#

mlab.close(all=True)
pixelsize=0.0196
terrain=readTerrain(name="bumps1",pixelsize=pixelsize,max_height=1.5)
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,mask=None,fig=fig, fontsize=0.3, details=2)
mlab.savefig("bumps1.png", figure=fig)

mlab.close(all=True)
pixelsize=0.0196
terrain=readTerrain(name="holes1",pixelsize=pixelsize,max_height=1.0)
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,mask=None,fig=fig, fontsize=0.3, details=2)
mlab.savefig("holes1.png", figure=fig)

mlab.close(all=True)
pixelsize=0.0196
terrain=readTerrain(name="slope_rocks1",pixelsize=pixelsize,max_height=1.0)
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,mask=None,fig=fig, fontsize=0.3, details=2)
mlab.savefig("slope_rock21.png", figure=fig)

mlab.close(all=True)
pixelsize=0.0196
terrain=readTerrain(name="steps1",pixelsize=pixelsize,max_height=1.2)
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,mask=None,fig=fig, fontsize=0.3, details=2)
mlab.savefig("steps1.png", figure=fig)

mlab.close(all=True)
pixelsize=0.0196
terrain=readTerrain(name="rails1",pixelsize=pixelsize,max_height=1.0)
fig=mlab.figure(size=(1000,1000))
visualizeTerrain(terrain,pixelsize,mask=None,fig=fig, fontsize=0.3, details=2)
mlab.savefig("rails1.png", figure=fig)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# visualize patches (samples) from a training file
#

pixelsize=0.02
# for no robot use robot_pose = (-1000,-1000,-1000,-1000,-1000)
no_robot = (-1000,-1000,-1000,-1000,-1000)
# trues
terrain_ct = skimage.io.imread_collection("inputs/sample_patches/t/*.png");
i=1
mlab.close(all=True)
for terrain in terrain_ct:
    terrain=terrain.astype(float)/255*1.0
    terrain=terrain-np.nanmin(terrain)
    #mlab.options.offscreen = True # To avoid opening a window
    fig=mlab.figure(size=(1000,1000))
    visualizeTerrainPatch(terrain,pixelsize,fig=fig,fontsize=0.1,robot_pose=(0.6,0.6,0.2,0.5,90), stext=False, border_color=(0.2,1.0,0.2))
    mlab.savefig("sample_3d_t_"+str(i)+".png",figure=fig)
    #saveAnimation(fig, "terrain-360", motion="360")
    #saveAnimation(fig, "terrain-osc", motion="oscillate")
    mlab.close(fig)
    i = i+ 1
#negatives
terrain_cnt = skimage.io.imread_collection("inputs/sample_patches/nt/*.png");
i=1
mlab.close(all=True)
for terrain in terrain_cnt:
    terrain=terrain.astype(float)/255*1.0
    terrain=terrain-np.nanmin(terrain)
    #mlab.options.offscreen = True # To avoid opening a window
    fig=mlab.figure(size=(1000,1000))
    visualizeTerrainPatch(terrain,pixelsize,fig=fig,fontsize=0.1,robot_pose=(0.6,0.6,0.2,0.5,90), stext=False, border_color=(1.0,0.2,0.2))
    mlab.savefig("sample_3d_nt_"+str(i)+".png",figure=fig)
    #saveAnimation(fig, "terrain-360", motion="360")
    #saveAnimation(fig, "terrain-osc", motion="oscillate")
    mlab.close(fig)
    i = i+ 1
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# visualize the paths from the pareto front (from yamls files generated by 
# the plan generator)
#

# display pioneer's model on a position (for the moment orientation is not taken into account)
#
pixelsize=0.02
stride = 9
offset = [30,30] # (px) graph was not constructued starting from (0,0) but from (patch_sze/2, patchsize/2) so we need to consider this offset
mlab.close(all=True)
# robot pose is set manually, todo: use the paths origin and target to calculate robots pose
terrain = readTerrain("quarry",pixelsize=pixelsize)
fig=mlab.figure(size=(1000,1000))
r_x, r_y = [offset[0]*pixelsize+pixelsize*stride*43 ,offset[1]*pixelsize+pixelsize*stride*125]
visualizeTerrain(terrain,pixelsize,fig=fig,fontsize=0.0,robot_pose=(r_x,r_y,3.7,0.5,45))

#
# plot the quarry and then plot selected set of paths from yamls files
#
pixelsize=0.02
stride = 9
offset = [30,30] # (px) graph was not constructued starting from (0,0) but from (patch_sze/2, patchsize/2) so we need to consider this offset
mlab.close(all=True)
# robot pose is set manually, todo: use the paths origin and target to calculate robots pose
terrain = readTerrain("quarry",pixelsize=pixelsize)
fig=mlab.figure(size=(1000,1000))

visualizeTerrain(terrain,pixelsize,fig=fig,fontsize=0.0,robot_pose=(0,0,0,0,0))

r_x, r_y = [offset[0]*pixelsize+pixelsize*stride*101 ,offset[1]*pixelsize+pixelsize*stride*70]
visualizeOnlyPathsonTerrain(fig, terrain, "inputs/paths/quarry_pioneer_stride_9-0.95-0.1-17341-5313-sol.yaml", offset, pixelsize, stride, (r_x,r_y,1.5,0.5,45), 0.4, out_name = "quarry_stride_14-0.95-0.1-2208-7195-sol", list_paths=[1,18])

#visualizeOnlyPathsonTerrain(fig, terrain, "inputs/paths/quarry_pioneer_stride_9-0.95-0.1-13161-17341-sol.yaml", offset, pixelsize, stride, (r_x,r_y,4.7,0.8,230), 0.2, out_name = "quarry_stride_14-0.95-0.1-2208-7195-sol", list_paths=[])

r_x, r_y = [offset[0]*pixelsize+pixelsize*stride*124 ,offset[1]*pixelsize+pixelsize*stride*5]
visualizeOnlyPathsonTerrain(fig, terrain, "inputs/paths/quarry_pioneer_stride_9-0.95-0.1-21209-26482-sol.yaml", offset, pixelsize, stride, (r_x,r_y,0.7,0.5,250), 0.4, out_name = "quarry_stride_14-0.95-0.1-2208-7195-sol", list_paths=[2,6,3])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# visualize the reachability values for a specific starting point (if available )
#

pixelsize=0.02
stride = 14
offset = [30,30] # (px) graph was not constructued starting from (0,0) but from (patch_sze/2, patchsize/2) so we need to consider this offset
mlab.close(all=True)
# robot pose is set manually, todo: use the paths origin and target to calculate robots pose
terrain = readTerrain("quarry",pixelsize=pixelsize)
r_x, r_y = [offset[0]*pixelsize+pixelsize*stride*20 ,offset[1]*pixelsize+pixelsize*stride*8]
visualizeReachabilityonTerrain(terrain, "inputs/reachability/quarry_stride_14-0.95-(8, 20)-reach.yaml", offset, pixelsize, stride, (r_x,r_y,4.7,0.8,230), 0.2, out_name = "quarry_stride_14-0.95-(8, 20)-reach")

r_x, r_y = [offset[0]*pixelsize+pixelsize*stride*65 ,offset[1]*pixelsize+pixelsize*stride*45]
visualizeReachabilityonTerrain(terrain, "inputs/reachability/quarry_stride_14-0.95-(45, 65)-reach.yaml", offset, pixelsize, stride, (r_x,r_y,1.4,0.8,40), 0.2, out_name = "quarry_stride_14-0.95-(45, 65)-reach")

r_x, r_y = [offset[0]*pixelsize+pixelsize*stride*80 ,offset[1]*pixelsize+pixelsize*stride*5]
visualizeReachabilityonTerrain(terrain, "inputs/reachability/quarry_stride_14-0.95-(5, 80)-reach.yaml", offset, pixelsize, stride, (r_x,r_y,0.3,0.8,250), 0.2, out_name = "quarry_stride_14-0.95-(5, 80)-reach")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# For visualizing the reachability and the paths on the slope
#
#

def readTerrain(name,pixelsize=0.02):
    hm = skimage.io.imread(heightmap_png)
    print(hm.min(),hm.max())
    
    if hm.ndim > 2: #multiple channels
        hm=skimage.color.rgb2gray(hm) #rgb2gray does the averaging and channel reduction
    
    elif hm.ndim == 2: #already in one channel
        #this is mostly for the images treated in matlab beforehand (one channel + grayscale + 16bit)
        if hm.dtype == 'uint8':
            divided = 255.0
        if hm.dtype == 'uint16':
            divided = 65535.0
        hm=hm/divided
    
    hm = hm * height_scale_factor
    
    tpixelsize=pixelsize
    hm=skimage.transform.resize(hm,(np.array(hm.shape)*tpixelsize/pixelsize).astype(int))
    hm=hm-np.nanmin(hm)
    
    return hm

def create_mask_reachability(json_str, img_size, patch_size, stride):
    attrs = json.loads(json_str)
    mat = np.array([list(e) for e in attrs])
    mask = np.zeros((img_size[0],img_size[1]),dtype='float64') 
    (lx, ly) = np.shape(mat)
    print(np.shape(mat),lx,ly)
    pol_s = int(stride/2)+0.5
    for i in range(0,lx):
        for j in range(0,ly):
            px = int(patch_size/2)+(i*stride)
            py = int(patch_size/2)+(j*stride)
            cx = ( px - pol_s, px + pol_s, px + pol_s, px - pol_s, px - pol_s)
            cy = ( py - pol_s, py - pol_s, py + pol_s, py + pol_s, py - pol_s)
            rr, cc = skimage.draw.polygon( cx,cy, shape = mask.shape)
            mask[rr,cc] = mat[i,j]
    return mask


def rviz_to_mayavi(point, max_x, max_y):
    mx = max_y - point[0]
    my = max_x - point[1]
    mz = point[2]
    return (mx,my,mz)


def makeRobot2(x,y,z,length,angle=0,fontsize=0.4):
    vertices=np.array([[0,0,1],[0,1,0],[-3,0,0],[0,-1,0]]).astype(float)
    verticesp=np.copy(vertices)
    verticesp[:,0]=vertices[:,0]*math.cos(angle)-vertices[:,1]*math.sin(angle)
    verticesp[:,1]=vertices[:,1]*math.cos(angle)+vertices[:,0]*math.sin(angle)
    vertices = verticesp/3*length+np.array([x,y,z])
    triangles=[[0,1,3],[0,1,2],[0,3,2],[3,1,2]]
    
    text = mlab.text3d(vertices[2,0]+0.1, vertices[2,1]+0.1, vertices[2,2], "robot", scale=fontsize, color=(0,0,0))
    
    # plot the stol file of the pioneer p3at robot
    STLfile="pioneer3at.stl"
    f=open(STLfile,'r')
    
    x=[]
    y=[]
    z=[]
    
    for line in f:
    	strarray=line.split()
    	if strarray[0]=='vertex':
    		x=np.append(x,np.double(strarray[1])*length)
    		y=np.append(y,np.double(strarray[2])*length)
    		z=np.append(z,np.double(strarray[3])*length)
    
    triangles=[(i, i+1, i+2) for i in range(0, len(x),3)]
    
    mlab.triangular_mesh(vertices[0,0]+  x*math.cos(math.radians(angle))-y*math.sin(math.radians(angle))  , vertices[0,1]+ y*math.cos(math.radians(angle))+x*math.sin(math.radians(angle)) , vertices[0,2]+z, triangles, color=(0.2,0.2,1.0))
    mlab.show()
    
    return()


def visualizeTerrainOpt(terrain, pixelsize, mask, fig=None, fontsize=0.4, robot_pose=(-1,0,0,0.8,0)):
    """Draws a surface colored according to mask using a custom colormap
    Inspired by http://gael-varoquaux.info/programming/mayavi-representing-an-additional-scalar-on-surfaces.html """
    
    if(fig is None):
        fig = mlab.figure()
    
    fig.scene.background = (1,1,1)
    
    y, x = np.meshgrid(np.arange(terrain.shape[0])*pixelsize,np.arange(terrain.shape[1])*pixelsize)
    s = mlab.mesh(x, y, terrain, scalars=mask) # We use a mesh, not a surf. Note: it's less efficient
    
    colormap=np.tile(np.array([180,180,180,255]),[256,1]) # all gray
    colormap[:,0]=np.linspace(180,100,256)
    colormap[:,1]=np.linspace(180,190,256)
    colormap[:,2]=np.linspace(180,255,256)

    s.module_manager.scalar_lut_manager.lut.table = colormap
    s.module_manager.scalar_lut_manager.lut.range=np.array([0.0,1.0])
    s.actor.property.interpolation = 'phong'
    s.actor.property.specular = 0.0
    s.actor.property.specular_power = 10
    s.actor.property.ambient_color = (1,1,1)

    s.actor.property.ambient = 0.02
    
    mlab.colorbar(s, title=None, orientation='vertical', nb_labels=3, nb_colors=None, label_fmt='%.1f')
    
    square=np.array([[0,0],[1,0],[1,1],[0,1]])[[0,1,2,3,0],:]
    square=np.hstack((square*np.array([[np.max(x),np.max(y)]]), np.zeros((5,1))))
    base=mlab.plot3d(square[:,0], square[:,1], square[:,2], color=(0,0,0), line_width=2)
    for i in range(4):
        p=np.mean(square[[i,i+1],:],axis=0)
        d=np.linalg.norm(square[i+1,:]-square[i+0,:])
        mlab.text3d(p[0], p[1], p[2], "{:.1f}m".format(d), scale=fontsize, color=(0,0,0))
    height=mlab.plot3d(np.array([0.0,0.0]), np.array([0.0,0.0]), np.array([0.0,np.nanmax(terrain)]), color=(0,0,0), line_width=2)
    mlab.text3d(0.0, 0.0, np.nanmax(terrain)*1.0, "{:.1f}m".format(np.nanmax(terrain)), scale=fontsize, color=(0,0,0))
    
    makeRobot2(robot_pose[0],robot_pose[1],robot_pose[2],robot_pose[3], angle=robot_pose[4], fontsize=fontsize)
        
    from tvtk.api import tvtk
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()


###

heightmap_png = "inputs/idsia_outdoors_smoothed7.png"

height_scale_factor = 2.5
pixel_size = 0.02

im = readTerrain(heightmap_png, pixel_size)

map_max_x = 3.56
map_max_y = 3.56

mask_alpha=skimage.io.imread("inputs/mask_alpha.png")[:,:,3].astype(float)/255
im = im*mask_alpha
im[im == 0] = np.nan

net_data = nx.read_graphml("paths/Slope_reach_path.graphml")

attributes_reach = nx.get_node_attributes(net_data, 'reach')
attributes_position = nx.get_node_attributes(net_data, 'position')

node = '1'
mask = create_mask_reachability(attributes_reach[node], im.shape, 10, 2)

skimage.io.imshow(mask)
attrs = json.loads(attributes_position[node])
source = np.array([float(e) for e in attrs])
rpos = rviz_to_mayavi((source[0]*0.398876404,source[1]*0.398876404,0),map_max_x,map_max_y )
print(source[0],source[1]," > ",  rpos)
visualizeTerrain3(im, pixel_size, mask=mask, fontsize=0.2, robot_pose=(rpos[0],rpos[1],0.4,0.7,90))

visualizePathsonTerrainOpt(im, net_data, node, map_max_x, map_max_y, pixelsize = pixel_size, stride = 2, fontsize=0.2, robot_pose=(rpos[0],rpos[1],0.4,0.8,90), mask=mask)



