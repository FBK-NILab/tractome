# -*- coding: utf-8 -*-

"""This is the part that connects the logic of the tractome
functionalities to the GUI.

Copyright (c) 2012-2014, Emanuele Olivetti and Eleftherios Garyfallidis
Distributed under the BSD 3-clause license. See COPYING.txt.
"""


import numpy as np

# fos modules
from fos import Actor
from fos.modelmat import screen_to_model
import fos.interact.collision as cll
from fos.coords import img_to_ras_coords, from_matvec

# pyglet module
from pyglet.gl import *
from ctypes import cast, c_int, POINTER

# dipy modules
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle
from dipy.viz.colormap import orient2rgb
from dipy.tracking.vox2track import track_counts

# other
import copy 
import cPickle as pickle

# Tk dialogs
import Tkinter, tkFileDialog

# Pyside for windowing
from PySide.QtCore import Qt
# Interaction Logic:
from manipulator import Manipulator

from itertools import chain

import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree

question_message = """
>>>>Track Labeler
P : select/unselect the representative track.
E : expand/collapse the selected streamlines 
F : keep selected streamlines rerun QuickBundles and hide everything else.
A : select all representative streamlines which are currently visible.
I : invert selected streamlines to unselected
H : hide/show all representative streamlines.
>>>Mouse
Left Button: keep pressed with dragging - rotation
Scrolling :  zoom
Shift + Scrolling : fast zoom
Right Button : panning - translation
Shift + Right Button : fast panning - translation
>>>General
F1 : Fullscreen.
F2 : Next time frame.
F3 : Previous time frame.
F4 : Automatic rotation.
F12 : Reset camera.
ESC: Exit.
? : Print this help information.
"""


def streamline2rgb(streamline):
    """Compute orientation of a streamline and retrieve and appropriate RGB
    color to represent it.
    """
    # simplest implementation:
    tmp = orient2rgb(streamline[0] - streamline[-1])
    return tmp

def apply_transformation(ijk, affine):
    """ Apply a 4x4 affine transformation

    Parameters
    ----------
    ijk : array, shape (N, 3)
        image coordinates
    affine : array, shape (4, 4)
        transformation matrix 

    Returns
    -------
    xyz : array, shape (N, 3)
        world coordinates in RAS (Neurological Convention)

    """

    ijk = ijk.T
    ijk1 = np.vstack((ijk, np.ones(ijk.shape[1])))
    xyz1 = np.dot(affine, ijk1)
    xyz = xyz1[:-1, :]
    return xyz.T


def compute_colors(streamlines, alpha):
    """Compute colors for a list of streamlines.
    """
    # assert(type(streamlines) == type([]))
    tot_vertices = np.sum([len(curve) for curve in streamlines])
    color = np.empty((tot_vertices,4), dtype='f4')
    counter = 0
    for curve in streamlines:
        color[counter:counter+len(curve),:3] = streamline2rgb(curve).astype('f4')
        counter += len(curve)
    color[:,3] = alpha
    return color


def compute_buffers(streamlines, alpha, save=False, filename=None):
    """Compute buffers for GL.
    """
    tmp = streamlines
    if type(tmp) is not type([]):
        tmp = streamlines.tolist()
    streamlines_buffer = np.ascontiguousarray(np.concatenate(tmp).astype('f4'))
    streamlines_colors = np.ascontiguousarray(compute_colors(streamlines, alpha))
    streamlines_count = np.ascontiguousarray(np.array([len(curve) for curve in streamlines],dtype='i4'))
    streamlines_first = np.ascontiguousarray(np.concatenate([[0],np.cumsum(streamlines_count)[:-1]]).astype('i4'))
    tmp = {'buffer': streamlines_buffer,
           'colors': streamlines_colors,
           'count': streamlines_count,
           'first': streamlines_first}
    if save:
        print "saving buffers to", filename
        np.savez_compressed(filename, **tmp)
    return tmp


def compute_buffers_representatives(buffers, representative_ids):
    """Compute OpenGL buffers for representatives from tractography
    buffers.
    """
    print "Creating buffers for representatives."
    count = buffers['count'][representative_ids].astype('i4')
    first = buffers['first'][representative_ids].astype('i4')
        
    representative_buffers = {'buffer': buffers['buffer'],
                              'colors': buffers['colors'].copy(),
                              'count': np.ascontiguousarray(count),
                              'first': np.ascontiguousarray(first)}
    return representative_buffers


def buffer2coordinates(buffer, first, count):
    """Extract an array of streamlines' coordinates from a buffer.

    This is meant mainly when the input 'buffers' is
    'representative_buffers'.
    """    
    return np.array([buffer[first[i]:first[i]+count[i]].astype(np.object) \
                     for i in range(len(first))])


def mbkm_wrapper(full_dissimilarity_matrix, n_clusters, streamlines_ids):
    """Wrapper of MBKM with API compatible to the Manipulator.

    streamlines_ids can be set or list.
    """
    sids = np.array(list(streamlines_ids))
    dissimilarity_matrix = full_dissimilarity_matrix[sids]

    print "MBKM clustering time:",
    init = 'random'
    mbkm = MiniBatchKMeans(init=init, n_clusters=n_clusters, batch_size=1000,
                          n_init=10, max_no_improvement=5, verbose=0)
    t0 = time.time()
    mbkm.fit(dissimilarity_matrix)
    t_mini_batch = time.time() - t0
    print t_mini_batch

    print "exhaustive smarter search of the medoids:",
    medoids_exhs = np.zeros(n_clusters, dtype=np.int)
    t0 = time.time()
    idxs = []
    for i, centroid in enumerate(mbkm.cluster_centers_):
        idx_i = np.where(mbkm.labels_==i)[0]
        if idx_i.size == 0: idx_i = [0]
        tmp = full_dissimilarity_matrix[idx_i] - centroid
        medoids_exhs[i] = sids[idx_i[(tmp * tmp).sum(1).argmin()]]
        idxs.append(set(sids[idx_i].tolist()))
        
    t_exhs_query = time.time() - t0
    print t_exhs_query, "sec"
    clusters = dict(zip(medoids_exhs, idxs))
    return clusters


class StreamlineLabeler(Actor, Manipulator):   
    """The Labeler for streamlines.
    """
    def __init__(self, name, buffers, clusters, representative_buffers=None, colors=None, vol_shape=None, representatives_line_width=5.0, streamlines_line_width=2.0, representatives_alpha=1.0, streamlines_alpha=1.0, affine=None, verbose=False, clustering_parameter=None, clustering_parameter_max=None, full_dissimilarity_matrix=None):
        """StreamlineLabeler is meant to explore and select subsets of
        the streamlines. The exploration occurs through clustering in
        order to simplify the scene.
        """
        # super(StreamlineLabeler, self).__init__(name)
        Actor.__init__(self, name) # direct call of the __init__ seems better in case of multiple inheritance

        if affine is None: self.affine = np.eye(4, dtype = np.float32)
        else: self.affine = affine
        if vol_shape is not None:
            I, J, K = vol_shape
            centershift = img_to_ras_coords(np.array([[I/2., J/2., K/2.]]), affine)
            centeraffine = from_matvec(np.eye(3), centershift.squeeze())
            affine[:3,3] = affine[:3, 3] - centeraffine[:3, 3]
        self.glaffine = (GLfloat * 16)(*tuple(affine.T.ravel()))
        self.glaff = affine
         
        self.mouse_x=None
        self.mouse_y=None

        self.buffers = buffers
        

        self.clusters = clusters
        self.save_init_set = True
         
        # MBKM:
        Manipulator.__init__(self, initial_clusters=clusters, clustering_function=mbkm_wrapper)

        # We keep the representative_ids as list to preserve order,
        # which is necessary for presentation purposes:
        self.representative_ids_ordered = sorted(self.clusters.keys())

        self.representatives_alpha = representatives_alpha

        # representative buffers:
        if representative_buffers is None:
            representative_buffers = compute_buffers_representatives(buffers, self.representative_ids_ordered)

        self.representatives_buffer = representative_buffers['buffer']
        self.representatives_colors = representative_buffers['colors']
        self.representatives_first = representative_buffers['first']
        self.representatives_count = representative_buffers['count']

        self.representatives = buffer2coordinates(self.representatives_buffer,
                                                  self.representatives_first,
                                                  self.representatives_count)

        # full tractography buffers:
        self.streamlines_buffer = buffers['buffer']
        self.streamlines_colors = buffers['colors']
        self.streamlines_first = buffers['first']
        self.streamlines_count = buffers['count']
        
        print('MBytes %f' % (self.streamlines_buffer.nbytes/2.**20,))

        self.hide_representatives = False
        self.expand = False
        self.expanded = False
        self.knnreset = False
        self.representatives_line_width = representatives_line_width
        self.streamlines_line_width = streamlines_line_width
        self.vertices = self.streamlines_buffer # this is apparently requested by Actor
        
        self.color_storage = {}
        # This is the color of a selected representative.
        self.color_selected = np.array([1.0, 1.0, 1.0, 1.0], dtype='f4')

        # This are the visualized streamlines.
        # (Note: maybe a copy is not strictly necessary here)
        self.streamlines_visualized_first = self.streamlines_first.copy()
        self.streamlines_visualized_count = self.streamlines_count.copy()
        
        # Clustering:
        self.clustering_parameter = clustering_parameter
        self.clustering_parameter_max = clustering_parameter_max
        self.full_dissimilarity_matrix = full_dissimilarity_matrix
        self.cantroi = 0
    
    def set_streamlines_ROIs(self, streamlines_rois_ids):
        """
        Set streamlines belonging to ROIs
        """

        if not hasattr(self, 'clusters_before_roi') or  len(self.clusters_before_roi)==0:
            self.clusters_before_roi = self.clusters
            
        self.streamlines_rois = streamlines_rois_ids
        if len(streamlines_rois_ids)>0:
            
            #1- Intersect ROIs based on the whole tractography with
            #actual clusters. From here I should obtain the "same"
            #clusters but only with streamlines from ROI.
            clusters_new = {}
            for rid in self.clusters_before_roi:
                new_cluster_ids = self.clusters_before_roi[rid].intersection(streamlines_rois_ids)
                if len(new_cluster_ids) > 0:
                    clusters_new[rid] = new_cluster_ids
                    clusters_new[list(new_cluster_ids)[0]] = clusters_new.pop(rid)
                    
            self.clusters_reset(clusters_new)
            self.recluster_action()
            self.hide_representatives = True
            self.select_all()
            self.expand = True
            
#
#            
#        else:
#            #Going back to show Clsuters before ROI was applied
#            # 1) sync self.representative_ids_ordered with original clusters before ROI:
#            self.representative_ids_ordered = sorted(self.clusters.keys())
#            # 2) change first and count buffers of representatives:
#            self.representatives_first = np.ascontiguousarray(self.streamlines_first[self.representative_ids_ordered], dtype='i4')
#            self.representatives_count = np.ascontiguousarray(self.streamlines_count[self.representative_ids_ordered], dtype='i4')
#            # 3) recompute self.representatives:
#            # (this is needed just for get_pointed_representative())
#            self.representatives = buffer2coordinates(self.representatives_buffer,
#                                                  self.representatives_first,
#                                                  self.representatives_count)
#            # 4) recompute self.streamlines_visualized_first/count:
#            streamlines_ids = list(reduce(chain, [self.clusters[rid] for rid in self.clusters]))
#            self.streamlines_visualized_first = np.ascontiguousarray(self.streamlines_first[streamlines_ids], dtype='i4')
#            self.streamlines_visualized_count = np.ascontiguousarray(self.streamlines_count[streamlines_ids], dtype='i4')
#            self.hide_representatives = False
#            self.expand = False
#            self.numstream_handler.fire(len(streamlines_ids))
#            self.numrep_handler.fire(len(representative_ids))

    def set_streamlines_knn(self,  streamlines_knn):
        """
        Set streamlines for KNN-extension
        """ 
        # 1) Saving the clusters available before the extension is done. In case the user goes back to k=0, we go directly to this stage.
        if self.save_init_set == True :
            self.clusters_before_knn = copy.deepcopy(self.clusters)
            # This KDTree is only computed on the medoids of clusters, for the assignment process. It is only computed once, unless the initial set of clusters changes and it is recomputed.
            self.kdtree_medoids= KDTree(self.full_dissimilarity_matrix[self.clusters.keys()])
            self.save_init_set = False
            
        clusters_new = copy.deepcopy(self.clusters_before_knn)
        clusters_representatives = self.clusters.keys()
        
        # 2) If the number of available clusters is 1, all neighbors will of course automatically be assigned to this cluster
        if len(clusters_representatives) == 1:
            clusters_new[clusters_representatives[0]].update(streamlines_knn)
        
        # 3) Query to previously computed KDTree, in order to find the nearest medoid (representative) of each streamline to be added.
        else:
            
            a2 = self.kdtree_medoids.query(self.full_dissimilarity_matrix[streamlines_knn],k=1, return_distance = False)
            for i in range(0, len(streamlines_knn)):
                clusters_new[clusters_representatives[a2[i, 0]]].add(streamlines_knn[i])

        self.clusters_reset(clusters_new)
        self.recluster_action()
        self.knnreset = True
        self.select_all()
        self.expand = True

    def set_empty_scene(self):
        """
        Hides all element in the screen if the ROI returns an empty set of streamlines
        """
        if not hasattr(self, 'clusters_before_roi') or  len(self.clusters_before_roi)==0:
            self.clusters_before_roi = self.clusters
        
        self.hide_representatives = True
        self.expand = False
         
    def reset_state(self,  function):
        """
        Show clustering state before any ROI or KNN-extension was applied 
        """
        if function =='roi':
            self.clusters_reset(self.clusters_before_roi)
            self.clusters_before_roi = {}
            self.recluster_action()

        elif function == 'knn':
            try:
                self.clusters_before_knn
                if self.save_init_set == True :
                    self.clusters_before_knn = copy.deepcopy(self.clusters)
                    # This KDTree is only computed on the medoids of clusters, for the assignment process. It is only computed once, unless the initial set of clusters changes and it is recomputed.
                    self.kdtree_medoids= KDTree(self.full_dissimilarity_matrix[self.clusters.keys()])
                    self.save_init_set = False
                self.clusters_reset(self.clusters_before_knn)
                self.recluster_action()
                self.select_all()
                self.expand = True
            except AttributeError:
                pass
           
        self.save_init_set = True
        self.hide_representatives = False
        
    
            
    def draw(self):
        """Draw virtual and real streamlines.

        This is done at every frame and therefore must be real fast.
        """
        
        glDisable(GL_LIGHTING)
        # representatives
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)        
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        # plot representatives if necessary:
        if not self.hide_representatives:
            glVertexPointer(3,GL_FLOAT,0,self.representatives_buffer.ctypes.data)
            glColorPointer(4,GL_FLOAT,0,self.representatives_colors.ctypes.data)
            glLineWidth(self.representatives_line_width)
            glPushMatrix()
            glMultMatrixf(self.glaffine)
            if isinstance(self.representatives_first, tuple): print '>> first Tuple'
            if isinstance(self.representatives_count, tuple): print '>> count Tuple'
            glMultiDrawArrays(GL_LINE_STRIP, 
                                    cast(self.representatives_first.ctypes.data,POINTER(c_int)), 
                                    cast(self.representatives_count.ctypes.data,POINTER(c_int)),  
                                    len(self.representatives_first))
            glPopMatrix()

        # plot tractography if necessary:
        if self.expand and len(self.selected) > 0:
            glVertexPointer(3,GL_FLOAT,0,self.streamlines_buffer.ctypes.data)
            glColorPointer(4,GL_FLOAT,0,self.streamlines_colors.ctypes.data)
            glLineWidth(self.streamlines_line_width)
            glPushMatrix()
            glMultMatrixf(self.glaffine)
            glMultiDrawArrays(GL_LINE_STRIP, 
                                    cast(self.streamlines_visualized_first.ctypes.data,POINTER(c_int)), 
                                    cast(self.streamlines_visualized_count.ctypes.data,POINTER(c_int)), 
                                    len(self.streamlines_visualized_first))
            glPopMatrix()
        
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)      
        glLineWidth(1.)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glEnable(GL_LIGHTING)


    # DO WE NEED THIS METHOD?
    def process_messages(self,messages):
        """
        """
        msg=messages['key_pressed']
        if msg!=None:
            self.process_keys(msg,None)
        msg=messages['mouse_position']            
        if msg!=None:
            self.process_mouse_position(*msg)


    def process_mouse_position(self, x, y):
        """
        """
        self.mouse_x = x
        self.mouse_y = y


    def process_keys(self, symbol, modifiers):
        """Bind actions to key press.
        """
        if symbol == Qt.Key_P:     
            rid = self.get_pointed_representative()
            print 'P : pick the representative pointed by the mouse =', rid
            self.select_toggle(rid)

        elif symbol == Qt.Key_A:
            print 'A: select all representatives.'
            self.select_all_toggle()

        elif symbol == Qt.Key_I:
            print 'I: invert selection of representatives.'
            self.invert()

        elif symbol == Qt.Key_H:
            print 'H: Hide/show representatives.'
            self.hide_representatives = not self.hide_representatives

        elif symbol == Qt.Key_E:
            print 'E: Expand/collapse streamlines of selected representatives.'
            self.expand_collapse_selected()

        elif symbol == Qt.Key_Backspace:
            print 'Backspace: Remove unselected representatives.'
            self.remove_unselected()
            self.save_init_set = True

        #elif symbol == Qt.Key_Delete:
           # print 'Delete: Remove selected representatives.'
            #self.remove_selected()

        elif symbol == Qt.Key_B:
            print "Go Back one step in the history."
            self.simple_history_back_one_step()
            
        elif symbol == Qt.Key_F:
            print "Go one step Forward in the history."
            self.simple_history_forward_one_step()
            
        elif symbol == Qt.Key_T:
            print 'T: Hide/show the whole tractography.'
            self.hide_representatives = not self.hide_representatives
            if self.hide_representatives :
                if self.expand :
                    self.expanded = True
                    self.expand = False

            if not self.hide_representatives and self.expanded:
                self.expand = True
                self.expanded = False


    def get_pointed_representative(self, min_dist=1e-3):
        """Compute the id of the closest streamline to the mouse pointer.
        """
        x, y = self.mouse_x, self.mouse_y
        # Define two points in model space from mouse+screen(=0) position and mouse+horizon(=1) position
        near = screen_to_model(x, y, 0)
        far = screen_to_model(x, y, 1)
        # Compute distance of representatives from screen and from the line defined by the two points above
        tmp = np.array([cll.mindistance_segment2track_info(near, far, apply_transformation(xyz, self.glaff)) \
                        for xyz in self.representatives])       
        line_distance, screen_distance = tmp[:,0], tmp[:,1]
        return self.representative_ids_ordered[np.argmin(line_distance + screen_distance)]
        
    
    def select_action(self, representative_id):
        """
        Steps for visualizing a selected representative.
        """
        print "select_action:", representative_id
        rid_position = self.representative_ids_ordered.index(representative_id)
        first = self.representatives_first[rid_position]
        count = self.representatives_count[rid_position]
        # this check is needed to let select_all_action() work,
        # otherwise a previously selected representative would be
        # stored as white and never get its original color back.
        if representative_id not in self.color_storage:
            self.color_storage[representative_id] = self.representatives_colors[first:first+count].copy() # .copy() is mandatory here otherwise that memory is changed by the next line!

        self.representatives_colors[first:first+count] = self.color_selected


    def unselect_action(self, representative_id):
        """Steps for visualizing an unselected representative.
        """
        print "unselect_action:", representative_id
        rid_position = self.representative_ids_ordered.index(representative_id)
        first = self.representatives_first[rid_position]
        count = self.representatives_count[rid_position]
        if representative_id in self.color_storage: # check to allow unselect_all_action()
            self.representatives_colors[first:first+count] = self.color_storage[representative_id]
            self.color_storage.pop(representative_id)


    def select_all_action(self):
        """
        """
        print "A: select all representatives."
        for rid in self.representative_ids_ordered:
            self.select_action(rid)


    def unselect_all_action(self):
        """
        """
        print "A: unselect all representatives."
        for rid in self.representative_ids_ordered:
            self.unselect_action(rid)
            

    def invert_action(self):
        """
        """
        print "I: invert selection of all representatives."
        for rid in self.representative_ids_ordered:
            if rid in self.selected:
                self.select_action(rid)
            else:
                self.unselect_action(rid)


    def expand_collapse_selected_action(self):
        """
        """
        print "E: Expand/collapse streamlines of selected representatives."
        if self.expand:
            print "Expand."
            if len(self.selected)>0:

                selected_streamlines_ids = list(reduce(chain, [self.clusters[rid] for rid in self.selected]))
                self.streamlines_visualized_first = np.ascontiguousarray(self.streamlines_first[selected_streamlines_ids], dtype='i4')
                self.streamlines_visualized_count = np.ascontiguousarray(self.streamlines_count[selected_streamlines_ids], dtype='i4')
        else:
            print "Collapse."
            
    
    def remove_unselected_action(self):
        """
        """
        print "Backspace: remove unselected."
        # Note: the following steps needs to be done in the given order.
        # 0) Restore original color to selected representatives.
        self.unselect_all()
        self.knnreset = False
        # 1) sync self.representative_ids_ordered with new clusters:
        self.representative_ids_ordered = sorted(self.clusters.keys())
        # 2) change first and count buffers of representatives:
        self.representatives_first = np.ascontiguousarray(self.streamlines_first[self.representative_ids_ordered], dtype='i4')
        self.representatives_count = np.ascontiguousarray(self.streamlines_count[self.representative_ids_ordered], dtype='i4')
        # 3) recompute self.representatives:
        # (this is needed just for get_pointed_representative())
        self.representatives = buffer2coordinates(self.representatives_buffer,
                                                  self.representatives_first,
                                                  self.representatives_count)
        # 4) recompute self.streamlines_visualized_first/count:
        streamlines_ids = list(reduce(chain, [self.clusters[rid] for rid in self.clusters]))
        self.streamlines_visualized_first = np.ascontiguousarray(self.streamlines_first[streamlines_ids], dtype='i4')
        self.streamlines_visualized_count = np.ascontiguousarray(self.streamlines_count[streamlines_ids], dtype='i4')
       

    def recluster_action(self):
        """
        """
        self.select_all()
        self.remove_unselected_action()
        self.knnreset = False



class ThresholdSelector(object):
    """
    """
    def __init__(self, parent, default_value, from_=1, to=500):
        """
        """
        self.parent = parent
        self.s = Tkinter.Scale(self.parent, from_=from_, to=to, width=25, length=300, orient=Tkinter.HORIZONTAL)
        self.s.set(default_value)
        self.s.pack()
        self.b = Tkinter.Button(self.parent, text='OK', command=self.ok)
        self.b.pack(side=Tkinter.BOTTOM)
    def ok(self):
        self.value = self.s.get()
        self.parent.destroy()



