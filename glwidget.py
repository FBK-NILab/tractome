"""

Copyright (c) 2012-2014, Emanuele Olivetti and Eleftherios Garyfallidis
Distributed under the BSD 3-clause license. See COPYING.txt.
"""

from PySide import QtCore, QtGui, QtOpenGL
from fos.world import *


try:
    from pyglet.gl import *
except ImportError:
    print("Need pyglet for OpenGL rendering")
    

empty_messages={ 'key_pressed':None,
                 'mouse_pressed':None,
                 'mouse_position':None,
                 'mouse_moved':None,
                 'mod_key_pressed':None}


class GLWidget(QtOpenGL.QGLWidget):
    """
    """
    def __init__(self, parent=None, 
                 width = None, 
                 height = None, 
                 bgcolor = None, 
                 enable_light = False, 
                 ortho = False):
        """
        """
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.lastPos = QtCore.QPoint()
        self.bgcolor = QtGui.QColor.fromRgbF(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
        self.width = width
        self.height = height
        self.enable_light = enable_light
        self.world = World()
        self.ortho = ortho
        self.messages = empty_messages
        self.setMouseTracking(True)
        # camera rotation speed
        self.ang_step = 0.02
        # necessary to grab key events
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.parent = parent


    def minimumSizeHint(self):
        """
        """
        return QtCore.QSize(50, 50)


    def sizeHint(self):
        """
        """
        return QtCore.QSize(self.width, self.height)


    def initializeGL(self):
        """
        """
        self.qglClearColor(self.bgcolor)
        glShadeModel(GL_SMOOTH)
        # glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if self.enable_light:
            self.world.setup_light()


    def paintGL(self):
        """
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self.world.draw_all()


    def resizeGL(self, width, height):
        """
        """
        # side = min(width, height)
        # glViewport((width - side) / 2, (height - side) / 2, side, side)
        if height == 0:
            height = 1
        vsml.setSize( width, height )
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        if self.ortho:
            self.width_ortho = self.width
            self.height_ortho = self.height
        self.update_projection()


    def update_projection(self, factor = 1.0):
        """
        """
        vsml.loadIdentity(vsml.MatrixTypes.PROJECTION)
        ratio =  abs(self.width * 1.0 / self.height)
        if self.ortho:
            self.width_ortho += -factor * ratio
            self.height_ortho += -factor
            vsml.ortho(-(self.width_ortho)/2.,(self.width_ortho)/2.,
                (self.height_ortho)/2.,(self.height_ortho)/-2.,-500,8000)
        else:
            vsml.perspective(60., ratio, .1, 8000)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(vsml.get_projection())
        glMatrixMode(GL_MODELVIEW)


    def ortho_zoom(self, zoom_level = 1.0):
        """
        """
        if not self.ortho:
            print('Not on orthogonal projection mode')
            return
        vsml.loadIdentity(vsml.MatrixTypes.PROJECTION)
        ratio =  abs(self.width * 1.0 / self.height)
        self.width_ortho = self.width * zoom_level * ratio
        self.height_ortho = self.width * zoom_level
        vsml.ortho(-(self.width_ortho)/2.,(self.width_ortho)/2.,
            (self.height_ortho)/2.,(self.height_ortho)/-2.,-500,8000)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(vsml.get_projection())
        glMatrixMode(GL_MODELVIEW)


    def mousePressEvent(self, event):
        """
        """
        self.lastPos = QtCore.QPoint(event.pos())
        if (event.modifiers() & QtCore.Qt.ControlModifier):
            x, y = event.x(), event.y()
            self.world.pick_all(x, self.height - y)


    def mouseMoveEvent(self, event):
        """
        """
        self.messages=empty_messages.copy()
        self.messages['mouse_position']=(event.x(),self.height - event.y())
        self.world.send_all_messages(self.messages)
        self.messages=empty_messages
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if (event.modifiers() & QtCore.Qt.ShiftModifier):
            shift = True
        else:
            shift = False
        if (event.modifiers() & QtCore.Qt.ControlModifier):
            ctrl = True
        else:
            ctrl = False
        if event.buttons() & QtCore.Qt.LeftButton:
            if not ctrl:
                # should rotate
                if dx != 0:
                    # rotate around yup
                    if dx > 0: angle = -self.ang_step #0.01
                    else: angle = self.ang_step #0.01
                    if shift: angle *= 2
                    self.world.camera.rotate_around_focal(angle, "yup")
                if dy != 0:
                    # rotate around right
                    if dy > 0: angle = -self.ang_step #0.01
                    else: angle = self.ang_step #0.01
                    if shift: angle *= 2
                    self.world.camera.rotate_around_focal(angle, "right")
                self.updateGL()
            else:
                # with control, do many selects!
                x, y = event.x(), event.y()
                self.world.pick_all( x, self.height - y)

        elif event.buttons() & QtCore.Qt.RightButton:
            # should pan
            if dx > 0: pandx = -1.0
            elif dx < 0: pandx = 1.0
            else: pandx = 0.0
            if dy > 0: pandy = 0.5
            elif dy < 0: pandy = -0.5
            else: pandy = 0.0
            if shift:
                pandx *= 4
                pandy *= 4
            self.world.camera.pan( pandx, pandy )
            self.updateGL()
            
        self.lastPos = QtCore.QPoint(event.pos())


    def wheelEvent(self, e):
        """
        """
        numSteps = e.delta() / 15 / 8
        if (e.modifiers() & QtCore.Qt.ControlModifier):
            ctrl = True
        else:
            ctrl = False
        if (e.modifiers() & QtCore.Qt.ShiftModifier):
            shift = True
        else:
            shift = False
        if self.ortho:
            if shift:
                numSteps *= 10
            self.update_projection( 10.*numSteps )
            self.updateGL()
            return
        if ctrl:
            if shift:
                self.world.camera.move_forward_all( numSteps * 10 )
            else:
                self.world.camera.move_forward_all( numSteps )
        else:
            if shift:
                self.world.camera.move_forward( numSteps * 10 )
            else:
                self.world.camera.move_forward( numSteps )
        self.updateGL()


    def keyPressEvent(self, event):
        """ Handle all key press events
        """
        key = event.key()
        self.messages=empty_messages.copy()
        self.messages['key_pressed']=key
        self.world.send_all_messages(self.messages)       
        self.updateGL()
        self.parent.keyPressEvent(event)
