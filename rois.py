# -*- coding: utf-8 -*-

"""

Copyright (c) 2012-2014, Emanuele Olivetti and Eleftherios Garyfallidis
Distributed under the BSD 3-clause license. See COPYING.txt.
"""

from fos import Actor
from pyglet.gl import *
from pyglet.lib import load_library # trick for the bug of pyglet arrays
glib = load_library('GL')
from fos.actor.primitives import *
from pylab import cm
import numpy as np
from fos.coords import img_to_ras_coords, from_matvec
from dipy.tracking import utils
## trick for the bug of pyglet multiarrays


class SphereTractome(Actor):
    """
    The actor for the ROI sphere
    """
    def __init__(self, name, x, y, z, radius, color, colorname,  method, coords, indextracks, affine, vol_shape):
        """
        """
        Actor.__init__(self, name) 
        self.coordinates = [x, y, z]
        self.radius = radius
        self.color=color
        self.colorname =colorname
        self.dims = vol_shape
        
        self.methods = {0 : self.trackvis,
           1 : self.tractome_inside,
           2 : self.tractome_intersect,
        }
        self.activemethod = self.methods[method]
        self.coords = coords
        self. indextracks  = indextracks
        # Generating index
        if affine is None: self.affine = np.eye(4, dtype = np.float32)
        else: self.affine = affine
        if method ==0:
            self.affine[1, 1] = self.affine[1, 1]*(-1)
        if vol_shape is not None:
            I, J, K = vol_shape
            centershift = img_to_ras_coords(np.array([[I/2., J/2., K/2.]]), affine)
            centeraffine = from_matvec(np.eye(3), centershift.squeeze())
            affine[:3,3] = affine[:3, 3] - centeraffine[:3, 3]
        self.glaffine = (GLfloat * 16)(*tuple(self.affine.T.ravel()))
        
        self.activemethod() 
       # vertices are still needed for something
        self.vertices = np.array([self.coordinates])


    def update_xcoord(self, coord):
        """
        """
        self.coordinates[0] = coord
        self.activemethod() 


    def update_ycoord(self, coord):
        """
        """
        self.coordinates[1] = coord
        self.activemethod() 


    def update_zcoord(self, coord):
        """
        """
        self.coordinates[2] = coord
        self.activemethod() 


    def update_radius(self, radius):
        """
        """
        self.radius = radius
        self.activemethod() 


    def update_color(self, color):
        """
        """
        self.color = color


    def update_method(self,  method,  coordsin= None,  index = None):
        """
        """
        self.coords = coordsin
        self.indextracks = index
        self.activemethod = self.methods[method]
        self.activemethod() 


    def trackvis(self):
        """
        Computing ROI that reproduces Trackvis results. Point is
        assumed to be in middle of the voxel and voxel 1 for them has
        index 1 and for us index 0.
        """
        voxel = [self.coordinates[0]  - 0.5, self.coordinates[1]  - 0.5,  self.coordinates[2]  - 0.5]
        #Convert coords to lps first
        self.coords[:,1] = self.dims[1] - self.coords[:,1]

        tmp = self.coords[:,0] - voxel[0]
        idx = np.where((tmp <= self.radius) & (tmp >=-self.radius))[0]
        tmp = self.coords[:,1][idx] - voxel[1]
        idx = idx[np.where((tmp <= self.radius) & (tmp >=-self.radius))[0]]
        tmp = self.coords[:,2][idx] - voxel[2]
        idx = idx[np.where((tmp <= self.radius) & (tmp >=-self.radius))[0]]
        tmp = self.coords[idx] - voxel
        idx = idx[np.where(np.sum((tmp * tmp), axis=1) <= (self.radius * self.radius))[0]]
        self.streamlines = np.unique(self.indextracks[idx])
    
        
    def tractome_inside(self):
        """
        Finds those streamlines that are inside the sphere defined by
        a center and radius.
        """
        tmp = self.coords[:,0] - self.coordinates[0]
        idx = np.where((tmp <= self.radius) & (tmp >=-self.radius))[0]
        tmp = self.coords[:,1][idx] - self.coordinates[1]
        idx = idx[np.where((tmp <= self.radius) & (tmp >=-self.radius))[0]]
        tmp = self.coords[:,2][idx] - self.coordinates[2]
        idx = idx[np.where((tmp <= self.radius) & (tmp >=-self.radius))[0]]
        tmp = self.coords[idx] - self.coordinates
        idx = idx[np.where(np.sum((tmp * tmp), axis=1) <= (self.radius * self.radius))[0]]
        self.streamlines = np.unique(self.indextracks[idx])


    def tractome_intersect(self):
        """
        """
        # TODO: not implemented yet
        raise NotImplementedError


    def draw(self):
        """
        """
        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPushMatrix()
        glColor4f(self.color[0], self.color[1], self.color[2], self.color[3])
        glMultMatrixf(self.glaffine)
        glTranslatef(self.coordinates[0], self.coordinates[1], self.coordinates[2])   
        sphere = gluNewQuadric()
        gluSphere(sphere, self.radius, 100, 100) 
        glPopMatrix()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        
        
        
class Mask(Actor):
    """
    Actor for loading a mask.
    """
    def __init__(self, name, color, colorname, coords, mask_coords, indextracks, affine, vol_shape):
        """
        """
        Actor.__init__(self, name) 
        self.color=color
        self.colorname =colorname

        # color_array = np.array(self.color*len(mask_coords[0]), dtype='f4')

        self.coords = coords
        self. indextracks  = indextracks
        self.mask_coords = mask_coords
        colors = [color,]*len(self.mask_coords[0])
        self.color_points = np.array(colors, dtype=np.float32)
        self.streamlines_mask()
        # Generating index
        
        if affine is None: self.affine = np.eye(4, dtype = np.float32)
        else: self.affine = affine
        if vol_shape is not None:
            I, J, K = vol_shape
            centershift = img_to_ras_coords(np.array([[I/2., J/2., K/2.]]), affine)
            centeraffine = from_matvec(np.eye(3), centershift.squeeze())
            affine[:3,3] = affine[:3, 3] - centeraffine[:3, 3]
        self.glaffine = (GLfloat * 16)(*tuple(self.affine.T.ravel()))
        
        # vertices are still needed for something
        self.vertices = np.array([self.voxels])


    def streamlines_mask(self):
        """
        Computing ROI that reproduces Trackvis results. 
        """
        
        streamlines = []
        voxels = []
        for i in range(len(self.mask_coords[0])):
            voxel = [self.mask_coords[0][i], self.mask_coords[1][i],  self.mask_coords[2][i]]
            voxels.append(voxel)
            tmp = np.array(self.coords[:,0])
            idx = np.where((voxel[0] <= tmp) & (tmp<= voxel[0]+1))[0]
            tmp = self.coords[:,1][idx]
            idx = idx[np.where((voxel[1] <= tmp) & (tmp<= voxel[1]+1))[0]]
            tmp = self.coords[:,2][idx]
            idx = idx[np.where((voxel[2]<= tmp) & (tmp<= voxel[2]+1))[0]]
            streamlines.extend(self.indextracks[idx])
        self.voxels = np.array(voxels, dtype='f4')
        self.streamlines = np.unique(streamlines)


    def update_color(self, color):
        """
        """
        self.color = color


    def draw(self):
        """
        """
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)        
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,self.voxels.ctypes.data)
        # glColor4f(self.color[0], self.color[1], self.color[2], self.color[3])
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4,GL_FLOAT,0,self.color_points.ctypes.data)
        glPointSize(6.)
        glPushMatrix()
        glMultMatrixf(self.glaffine)
        glib.glDrawArrays(GL_POINTS, 
                          0, 
                          len(self.vertices))
        glPopMatrix()
        
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)      
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glEnable(GL_LIGHTING)
