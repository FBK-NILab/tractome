# -*- coding: utf-8 -*-

"""This is the part that connects the logic of the tractome
functionalities to the GUI.

Copyright (c) 2012-2014, Emanuele Olivetti and Eleftherios Garyfallidis
Distributed under the BSD 3-clause license. See COPYING.txt.
"""


import pyglet
debug = False
pyglet.options['debug_gl'] = debug
pyglet.options['debug_gl_trace'] = debug
pyglet.options['debug_gl_trace_args'] = debug
pyglet.options['debug_lib'] = debug
pyglet.options['debug_media'] = debug
pyglet.options['debug_trace'] = debug
pyglet.options['debug_trace_args'] = debug
pyglet.options['debug_trace_depth'] = 1
pyglet.options['debug_font'] = debug
pyglet.options['debug_x11'] = debug
pyglet.options['debug_trace'] = debug

import numpy as np
import nibabel as nib
from streamshow import StreamlineLabeler
from guillotine import Guillotine
from dipy.io.dpy import Dpy
import pickle
from streamshow import compute_buffers, mbkm_wrapper
from fos.coords import img_to_ras_coords
from fos.actor import *
from fos.world import *
from rois import *
from itertools import chain
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.metrics import length
from dissimilarity_common import compute_dissimilarity
from sklearn.neighbors import KDTree
import os

class Tractome(object):
    """
    """
    def __init__(self):
        """
        Initializing the class that contains that manipulates the
        scene, actors and all the logic of Tractome.
        """
        self.scene = Scene(scenename = 'Main Scene', activate_aabb = False)
        self.d_active_ROIS = {}
        self.list_ROIS = []
        self.list_oper_ROIS = []
     
     
    def loading_structural(self, structpath = None):
        """
        Loading structural data.
        """
        # load structural volume
        print "Loading structural information file"
        self.structpath = structpath
        self.img = nib.load(self.structpath)
        data = self.img.get_data()
        self.affine = self.img.get_affine()
        self.dims = data.shape[:3]
        
        # verifying if structural is color_fa created by TrackVis :: Diffusion Toolkit
        if data.dtype == [('R', '|u1'), ('G', '|u1'), ('B', '|u1')]:
            data1 = data.view((np.uint8, len(data.dtype.names)))
            data = data1
            del data1
            
        
        # Create the Guillotine object
        data = (np.interp(data, [data.min(), data.max()], [0, 255]))
        self.guil = Guillotine('Volume Slicer', data, np.copy(self.affine))
        self.scene.add_actor(self.guil) 
    
        
    def loading_full_tractograpy(self, tracpath=None):
        """
        Loading full tractography and creates StreamlineLabeler to
        show it all.
        """
        # load the tracks registered in MNI space
        self.tracpath=tracpath
        basename = os.path.basename(self.tracpath)
        tracks_basename, tracks_format = os.path.splitext(basename)
        
        if tracks_format == '.dpy': 
            
            dpr = Dpy(self.tracpath, 'r')
            print "Loading", self.tracpath
            self.T = dpr.read_tracks()
            dpr.close()
            self.T = np.array(self.T, dtype=np.object)

            
        elif tracks_format == '.trk': 
            streams, self.hdr = nib.trackvis.read(self.tracpath, points_space='voxel')
            print "Loading", self.tracpath
            self.T = np.empty([len(streams)], dtype=np.object)
            self.T[:] = [s[0] for s in streams]
         

        print "Removing short streamlines"
        self.T = np.array([t for t in self.T if length(t)>= 15],  dtype=np.object)
        
        tracks_directoryname = os.path.dirname(self.tracpath) + '/.temp/'
        general_info_filename = tracks_directoryname + tracks_basename + '.spa'
        
        
        # Check if there is the .spa file that contains all the
        # computed information from the tractography anyway and try to
        # load it
        try:
            print "Looking for general information file"
            self.load_info(general_info_filename)
                    
        except (IOError, KeyError):
            print "General information not found, recomputing buffers"
            self.update_info(general_info_filename)
                    
        # create the interaction system for tracks, 
        self.streamlab  = StreamlineLabeler('Bundle Picker',
                                            self.buffers, self.clusters,
                                            vol_shape=self.dims, 
                                            affine=np.copy(self.affine),
                                            clustering_parameter=len(self.clusters),
                                            clustering_parameter_max=len(self.clusters),
                                            full_dissimilarity_matrix=self.full_dissimilarity_matrix)
                
        self.scene.add_actor(self.streamlab)


    def load_segmentation(self, segpath=None):
        """
        Loading file containing a previous segmentation
        """
        print "Loading saved session file"
        segm_info = pickle.load(open(segpath)) 
        state = segm_info['segmsession']  
            
        self.structpath=segm_info['structfilename']
        self.tracpath=segm_info['tractfilename']   

        # load T1 volume registered in MNI space
        print "Loading structural information file"

        self.loading_structural(self.structpath)

        # load tractography
        self.loading_full_tractograpy(self.tracpath)
        self.streamlab.set_state(state)
            
        self.scene.update()


    def max_num_clusters(self):
        """
        """
        n_clusters = len(self.streamlab.streamline_ids)
        if (len(self.streamlab.streamline_ids) < 1e5) and (len(self.streamlab.streamline_ids)>= 50):
            default = 50
        else:
            default = len(self.streamlab.streamline_ids)
            
        return n_clusters,  default


    def recluster(self,  n_clusters):
        """
        Re-cluster current selected set of streamlines
        """
        # MBKM:
        self.streamlab.recluster(n_clusters, data=self.full_dissimilarity_matrix)
        self.set_streamlines_clusters()
        


    def loading_mask(self,  filename,  color):
        """
        Loads a mask
        """
        print "Loading mask"

        img = nib.load(filename)
        mask = img.get_data()
        itemindex = np.where(mask!=0)
        
        self.ROIMask(os.path.basename(filename), itemindex, color)


    def ROIMask(self, nameroi,  mask,  color):
        """
        Create actor for ROI from loaded mask and add it to the scene.
        """
        coords_streamlines, index_streamlines = self.compute_dataforROI()
        
        self.list_ROIS.append(nameroi)
        self.d_active_ROIS[nameroi] = False
        self.list_oper_ROIS.append('and')
        
        #create Mask actor and add it to scene
        mask = Mask(nameroi, color.getRgbF(), color.name(), coords_streamlines, mask, index_streamlines, self.affine, self.dims)
        self.scene.add_actor(mask)


    def max_coordinates(self):
        """
        Computing maximum value of each coordinate from the whole
        tractography.
        """
        max= [np.amax(t,axis=0).tolist()  for t in self.T]
        coords_max = np.amax(max,axis=0)
         
        return coords_max


    def save_info(self, filepath):
        """
        Saves all the information from the tractography required for
        the whole segmentation procedure.
        """
        info = {'initclusters':self.clusters, 'buff':self.buffers, 'dismatrix':self.full_dissimilarity_matrix,'nprot':self.num_prototypes,  'kdtree':self.kdt}
        print "Saving information of the tractography for the segmentation"
        print filepath
        filedir = os.path.dirname(filepath)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        pickle.dump(info, open(filepath,'w'), protocol=pickle.HIGHEST_PROTOCOL)


    def load_info(self, filepath):
        """
        Loads all the information from the tractography required for
        the whole segmentation procedure.
        """
        print "Loading general information file"
        general_info = pickle.load(open(filepath))
        self.full_dissimilarity_matrix = general_info['dismatrix']
        self.num_prototypes = general_info['nprot']
        self.buffers = general_info['buff']
        self.clusters = general_info['initclusters']
        self.kdt = general_info['kdtree']

 
    def update_info(self, filepath):
        """
        Compute missing or inconsistent information in the cache.
        """
        save = False

        try:
            self.buffers
        except AttributeError:
            print "Computing buffers."
            self.buffers = compute_buffers(self.T, alpha=1.0, save=False)
            save = True
            
        try:
            self.num_prototypes
        except AttributeError:
            print "Defining number of prototypes"
            self.num_prototypes = 40
            save = True
            
        try:
            self.full_dissimilarity_matrix
        except AttributeError:
            print "Computing dissimilarity matrix"
            self.full_dissimilarity_matrix = compute_dissimilarity(self.T, distance=bundles_distances_mam, prototype_policy='sff', num_prototypes=self.num_prototypes)
            save = True    
        
        try:
            assert(self.full_dissimilarity_matrix.shape[0] == len(self.T))
        except AssertionError:
            print "Re-computing dissimilarity matrix."
            self.num_prototypes = 40
            self.full_dissimilarity_matrix = compute_dissimilarity(self.T, distance=bundles_distances_mam, prototype_policy='sff', num_prototypes=self.num_prototypes)
            save = True

        try:
            self.clusters
        except AttributeError:
            print "Computing MBKM"
            size_T = len(self.T)
            if  size_T > 150:
                n_clusters = 150
            else:
                n_clusters = size_T

            streamlines_ids = np.arange(size_T, dtype=np.int)
            self.clusters = mbkm_wrapper(self.full_dissimilarity_matrix, n_clusters, streamlines_ids)
            save = True
            
        try:
            self.kdt
        except AttributeError:
            print "Computing KDTree"
            self.compute_kdtree()
            save=True

        if save: self.save_info(filepath)


    def save_segmentation(self, filename):
        """
        Saves the information of the segmentation result from the
        current session.
        """
        
        print "Save segmentation result from current session"
        filename = filename[0]+'.seg'
        state = self.streamlab.get_state()
        seg_info={'structfilename':self.structpath, 'tractfilename':self.tracpath, 'segmsession':state}
        pickle.dump(seg_info, open(filename,'w'), protocol=pickle.HIGHEST_PROTOCOL)


    def save_trk(self, filename):
        """
        Save current streamlines in .trk file.
        """
        filename = filename[0]+'.trk'
        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = self.img.get_header().get_zooms()[:3]
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = self.dims
        hdr['vox_to_ras'] = self.affine

        streamlines_ids = list(self.streamlab.streamline_ids)
        streamlines = [(s,  None,  None) for s in self.T[streamlines_ids]]
        nib.trackvis.write(filename, streamlines, hdr, points_space = 'voxel')


    def compute_kdtree(self):
        """
        Compute kdtree from tactography, for ROIs and extensions.
        """
        self.kdt=KDTree(self.full_dissimilarity_matrix)
        
    
    def compute_kqueries(self,  k):
        """
        Makes the query to find the knn of the current streamlines on the scene
        """
       
        if k==0:
            if self.streamlab.knnreset == True:
                self.streamlab.reset_state('knn')
            
        else:
            if len(self.streamlab.streamline_ids) == len(self.T):
        
                raise TractomeError("Cannot enlarge clusters. The whole tractography is being used as input.")
   
            else:
                nn = k+1
                if self.streamlab.save_init_set==True:
                    self.streamlines_before_knn = self.streamlab.streamline_ids.copy()
                    
                a2 = self.kdt.query(self.full_dissimilarity_matrix[list( self.streamlines_before_knn)],k=nn, return_distance = False)
                b2 = set(a2.flat)
                
                # Finding difference between the initial set of streamlines and those from the kdt query. This will give us the new streamlines
                b2.difference_update(self.streamlines_before_knn)
                if len(b2)>0:
                    self.streamlab.set_streamlines_knn(list(b2))
                    
    
    def set_streamlines_clusters(self):
        """
        The actual composition of clusters will be set as reference to compute the new neighbors.
        """
        self.streamlab.save_init_set = True
        self.streamlab.hide_representatives = False


    def compute_dataforROI(self):
        """
        Compute info from tractography to provide it to ROI.
        """
        coords = np.vstack(self.T)
        index = np.concatenate([i*np.ones(len(s)) for i,s in enumerate(self.T)]).astype(np.int) 
        return coords,  index


    def create_ROI_sphere(self, nameroi,  coordx, coordy, coordz, radius, method, color,  colorname):
        """
        Create actor for ROI sphere and add it to the scene.
        """
        coords_streamlines, index_streamlines = self.compute_dataforROI()
        
        self.list_ROIS.append(nameroi)
        self.d_active_ROIS[nameroi] = False
        self.list_oper_ROIS.append('and')
        
        #create Sphere actor and add it to scene
        sphere = SphereTractome(nameroi, coordx, coordy, coordz, radius,  color,  colorname,  method, coords_streamlines, index_streamlines, self.affine, self.dims)
        
        self.scene.add_actor(sphere)
         
        
    def update_ROI(self, nameroi, newname = None,  coordx=None, coordy=None, coordz=None, radius=None, color=None,  method = None,  rebuild = False,  pos_activeroi = None):
        """
        Updates any parameter of the specified ROI.
        """
 
        if coordx is not None:
            self.scene.actors[nameroi].update_xcoord(coordx)
            
        if coordy is not None:
            self.scene.actors[nameroi].update_ycoord(coordy)
            
        if coordz is not None:
            self.scene.actors[nameroi].update_zcoord(coordz)
            
        if radius is not None:
            self.scene.actors[nameroi].update_radius(radius)
            
        if color is not None:
            self.scene.actors[nameroi].update_color (color) 
            
        if method is not None:
            value_dict = self.d_active_ROIS[nameroi]
            del self.d_active_ROIS[nameroi]
            actor_roi = self.scene.actors[str(nameroi)]
            self.scene.remove_actor(str(nameroi))
            self.create_ROI_sphere(str(nameroi),  actor_roi.coordinates[0], actor_roi.coordinates[1], actor_roi.coordinates[2], actor_roi.radius, method, actor_roi.color, actor_roi.colorname)
            
        if rebuild:
            self.compute_streamlines_ROIS()
        
        if newname is not None:
            value_dict = self.d_active_ROIS[nameroi]
            del self.d_active_ROIS[nameroi]
            self.d_active_ROIS[newname] = value_dict
            self.list_ROIS[pos_activeroi] = newname 
            actor_roi = self.scene.actors[nameroi]
            actor_roi.name = newname
            self.scene.remove_actor(str(nameroi))
            self.scene.add_actor(actor_roi)
            
          

    def activation_ROIs(self, pos_activeroi,  activate, operator=None):
        """
        Activates or deactivates the specified ROI. In case it is
        activated, the operator to be applied is also specified.
        """
        self.d_active_ROIS[self.list_ROIS[pos_activeroi]] = activate
        if operator is not None:
            self.list_oper_ROIS[pos_activeroi] = operator


    def compute_streamlines_ROIS(self):
        """
        Obtain set of streamlines that pass through the specified
        ROIs.
        """
        streamlines_ROIs = []
        last_chkd = -1

        for pos in range(0, len(self.list_ROIS)):
            name_roi  = self.list_ROIS[pos]
            if self.d_active_ROIS[name_roi]:
                if last_chkd == -1:
                    streamlines_ROIs = set(self.scene.actors[name_roi].streamlines)
                else:
                    current_roi_streamlines = set(self.scene.actors[name_roi].streamlines)
                    if self.list_oper_ROIS[last_chkd] == 'and':
                        streamlines_ROIs.intersection_update(current_roi_streamlines)
                    elif self.list_oper_ROIS[last_chkd] == 'or':
                        streamlines_ROIs.update(current_roi_streamlines)
     
                last_chkd =pos
        
        if last_chkd == -1:
            self.streamlab.reset_state()
        else:
            if len(streamlines_ROIs) > 0:
                self.streamlab.set_streamlines_ROIs(streamlines_ROIs)
            else:
                self.streamlab.set_empty_scene()

    
    def information_from_ROI(self, name_roi):
        """
        Returns general information from the specified ROI.
        """
        roi = self.scene.actors[name_roi]
        xcoord = roi.coordinates[0]
        ycoord = roi.coordinates[1]
        zcoord = roi.coordinates[2] 
        radius = roi.radius
        color = roi.colorname
        return xcoord,  ycoord,  zcoord,  radius,  color


    def clear_all(self):
        """
        Actors of scene will be removed in order to load new ones.
        """
        self.d_active_ROIS = {}
        self.list_ROIS = []
        self.list_oper_ROIS = []
        self.scene.actors.clear()
        self.scene.update()


    def clear_actor(self,  name):
        """
        Specified actor will be removed from the scenename
        """
        self.scene.remove_actor(name)


    def show_hide_actor(self,  name, state):
        """
        Show or hide the specified actor
        """
        if state:
            self.scene.actors[name].show()
           
        else:
            self.scene.actors[name].hide()
            
        self.scene.update()



class TractomeError(Exception):
   def __init__(self, arg):
      self.args = arg
 

