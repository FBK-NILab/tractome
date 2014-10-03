# -*- coding: utf-8 -*-

"""This is the Volume Slicer Actor, to visualize a 3D volumetric image
of the head as slices.

Copyright (c) 2012-2014, Emanuele Olivetti and Eleftherios Garyfallidis
Distributed under the BSD 3-clause license. See COPYING.txt.
"""

import numpy as np
from fos import Window, Scene
from fos.actor.slicer import Slicer
from pyglet.gl import *
from fos.coords import rotation_matrix, from_matvec
from fos import Init, Run
from PySide.QtCore import Qt
import copy


class Guillotine(Slicer):
    """ Volume Slicer Actor

    Notes
    ------
    Coordinate Systems
    http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    http://www.slicer.org/slicerWiki/index.php/Coordinate_systems
    http://eeg.sourceforge.net/mri_orientation_notes.html

    """
    def __init__(self, name, data, affine, 
                    convention='RAS', look='anteriorz+'):
        """ Volume Slicer that supports medical conventions

        Parameters
        ----------
        name : str
        data : array, shape (X, Y, Z) or (X, Y, Z, 3) or (X, Y, Z, 4)
        affine : array, shape (4, 4)
        convention : str,
                'RAS' for neurological,
                'LAS' for radiological (default)
        look : str,
                'anteriorz+' look in the subject from the front 

        """

        data[np.isnan(data)] = 0
        data = np.interp(data, [data.min(), data.max()], [0, 255])
        data = data.astype(np.ubyte)
        
        """
        if convention == 'RAS' and look == 'anteriorz+':
            axis = np.array([1, 0, 0.])
            theta = -90. 
            post_mat = from_matvec(rotation_matrix(axis, theta))
            axis = np.array([0, 0, 1.])
            theta = -90. 
            post_mat = np.dot(
                        from_matvec(rotation_matrix(axis, theta)), 
                        post_mat)
        """
        post_mat = np.eye(4)
        super(Guillotine, self).__init__(name, data, affine, convention, post_mat)

    def right2left(self, step):
        if self.i + step < self.I:
            self.slice_i(self.i + step)
        else:
            self.slice_i(self.I - 1)

    def left2right(self, step):
        if self.i - step >= 0:
            self.slice_i(self.i - step)
        else:
            self.slice_i(0)

    def inferior2superior(self, step):
        if self.k + step < self.K:
            self.slice_k(self.k + step)
        else:
            self.slice_k(self.K - 1)

    def superior2inferior(self, step):
        if self.k - step >= 0:
            self.slice_k(self.k - step)
        else:
            self.slice_k(0)

    def anterior2posterior(self, step):
        if self.j + step < self.J:
            self.slice_j(self.j + step)
        else:
            self.slice_j(self.J - 1)

    def posterior2anterior(self, step):
        if self.j - step >= 0:
            self.slice_j(self.j - step)
        else:
            self.slice_j(0)

    def reset_slices(self):
        self.slice_i(self.I / 2)
        self.slice_j(self.J / 2)
        self.slice_k(self.K / 2)

    def slices_ijk(self, i, j, k):
        self.slice_i(i)
        self.slice_j(j)
        self.slice_k(k)
        
    def show_coronal(self, bool=True):
        self.show_k = bool

    def show_axial(self, bool=True):
        self.show_i = bool

    def show_saggital(self, bool=True):
        self.show_j = bool

    def show_all(self, bool=True):
        self.show_i = bool
        self.show_j = bool
        self.show_k = bool
    
    def process_messages(self, messages):
        msg=messages['key_pressed']
        #print 'Processing messages in actor', self.name, 
        #' key_press message ', msg
        if msg!=None:
            self.process_keys(msg,None)

    def process_keys(self, symbol, modifiers):
        """Bind actions to key press.
        """
        if symbol == Qt.Key_Left:     
            print 'Left'
            if self.i < self.data.shape[0]:
                self.slice_i(self.i+1)
            else:
                self.slice_i(0)

        if symbol == Qt.Key_Left:     
            print 'Left'
            if self.i < self.data.shape[0]:
                self.slice_i(self.i+1)
            else:
                self.slice_i(0)
        
        if symbol == Qt.Key_Right:     
            print 'Right'
            if self.i >=0:
                self.slice_i(self.i-1)
            else:
                self.slice_i(self.data.shape[0]-1)

        if symbol == Qt.Key_Up:
            print 'Superior'
            if self.k < self.data.shape[2]:
                self.slice_k(self.k+1)
            else:
                self.slice_k(0)

        if symbol == Qt.Key_Down:
            print 'Interior'
            if self.k >= 0:
                self.slice_k(self.k-1)
            else:
                self.slice_k(self.data.shape[2]-1)

        if symbol == Qt.Key_PageUp:
            print 'Anterior'
            if self.j < self.data.shape[1]:
                self.slice_j(self.j+1)
            else:
                self.slice_j(0)

        if symbol == Qt.Key_PageDown:
            print 'Posterior'
            if self.j >= 0:
                self.slice_j(self.j-1)
            else:
                self.slice_j(self.data.shape[1]-1)

        if symbol == Qt.Key_0:
            self.show_i = not self.show_i
            self.show_j = not self.show_j
            self.show_k = not self.show_k

        if symbol == Qt.Key_1:
            self.show_i = not self.show_i

        if symbol == Qt.Key_2:
            self.show_j = not self.show_j

        if symbol == Qt.Key_3:
            self.show_k = not self.show_k

#        if symbol == Qt.Key_R:
#            self.slice_i(self.I / 2)
#            self.slice_j(self.J / 2)
#            self.slice_k(self.K / 2)
            

def anteriorzplus(xyz):
    axis = np.array([1, 0, 0.])
    theta = -90. 
    post_mat = from_matvec(rotation_matrix(axis, theta))
    axis = np.array([0, 1, 0])
    theta = 180. 
    post_mat = np.dot(
                from_matvec(rotation_matrix(axis, theta)), 
                post_mat)

    return np.dot(post_mat[:3, :3], xyz.T).T

