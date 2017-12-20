import re
import os
import sys
import numpy as np
import nibabel as nib
from dipy.viz import actor, window, ui
from dipy.tracking.streamline import transform_streamlines
from bottle import run, post, request, response
from threading import Thread

global stream_actor, slave_actor, show_m, last_msg

port = sys.argv[1]
print 'PORT', port
some = 10000
#some = 10
wtitle = ''
if port == '8081':
    wtitle = "NN - Nearest Neighbour"
elif port == '8082':
    wtitle = "GM - Graph Matching"
elif port == '8083':
    wtitle = "LAP - Linear Assignment"
else:
    wtitle = "No title"
print 'WTITLE', wtitle

#src = '/Users/paolo/Downloads/562345_D10RETEST_b2k_csddet_sift2_FS2016_ioff.left.trk'
#src = '/Users/paolo/Downloads/eudx_ioff.left.trk'
#src = '/Users/paolo/Downloads/562345_D10RETEST_b2k_csddet_sift2_FS2016_ioff.left.trk'
#src = './Demo_Correspondence/sub-124422_size-100k_tract.trk'
src = './Demo_Correspondence/sub-627549_size-100k_tract.trk'

'''
t1w_src = './Demo_Correspondence/sub-124422_space-dwi_T1w.nii.gz'
t1w_img = nib.load(t1w_src)
t1w_data = t1w_img.get_data()
t1w_aff = t1w_img.affine
mean, std = t1w_data[t1w_data > 0].mean(), t1w_data[t1w_data > 0].std()
value_range = (mean - 0.5 * std, mean + 1.5 * std)
slice_actor = actor.slicer(t1w_data, t1w_aff, value_range)
slice_actor.display(73, None, None)
'''

t = nib.streamlines.load(src)
#tract = transform_streamlines(t.streamlines, np.linalg.inv(t.affine))
tract = [s for s in t.streamlines] ################
subset = np.random.choice(len(tract), some, replace=False)
subset.sort()
simple = [tract[s] for s in subset]

renderer = window.Renderer()
stream_actor = actor.line(simple)
renderer.set_camera(position=(-176.42, 118.52, 128.20),
                    focal_point=(113.30, 128.31, 76.56),
                    view_up=(0.18, 0.00, 0.98))
renderer.set_camera(position=(0., 0., 260.),
                    focal_point=(0., 0., 0.),
                    view_up=(0., 1., 0.))
renderer.add(stream_actor)
#renderer.add(slice_actor)

show_m = window.ShowManager(renderer, size=(600, 600))
show_m.initialize()
show_m.window.SetWindowName(wtitle)
show_m.render()
ac = renderer.GetActiveCamera()
print ac.GetPosition()
print ac.GetFocalPoint()
print ac.GetViewUp()

#show_m.start()

@post('/process')
def my_process():
    global slave_actor, stream_actor, last_msg
    req_event = request.body.read()
    print req_event
    if 'Expand' in req_event:
        expand = [int(s) for s in req_event.strip('Expand ').split()]
        #expand = [0, 1, 2, 3, 4, 5,]
        etract = np.take(tract, expand)
        slave_actor = actor.line(etract)
        show_m.ren.add(slave_actor)
        stream_actor.SetVisibility(0)
        show_m.render()
    elif 'Collapse' in req_event:
        slave_actor.SetVisibility(0)
        show_m.ren.RemoveActor(slave_actor)
        stream_actor.SetVisibility(1)
        show_m.render()
    elif 'RefocusCameraView' in req_event:
        show_m.ren.RemoveActor(stream_actor)
        stream_actor = actor.line(simple)
        show_m.ren.add(stream_actor)
        stream_actor.SetVisibility(1)
        ac = show_m.ren.GetActiveCamera()
        print ac.GetPosition()
        print ac.GetFocalPoint()
        print ac.GetViewUp()
        ac.SetPosition(71.42986869812012, 87.12019729614258, 443.4783652957321)
        ac.SetFocalPoint(71.42986869812012, 87.12019729614258, 57.5451107025146)
        ac.SetViewUp(0., 1., 0.)
        show_m.render()
    elif 'SagittalLeftCameraView' in req_event:
        ac = show_m.ren.GetActiveCamera()
        print ac.GetPosition()
        print ac.GetFocalPoint()
        print ac.GetViewUp()
        ac.SetPosition(-470., -78., -78.)
        ac.SetFocalPoint(0.82, -17.12019729614258, 0.)
        ac.SetViewUp(0., 0., 1.)
        show_m.render()
    elif 'AxialTopCameraView' in req_event:
        ac = show_m.ren.GetActiveCamera()
        print ac.GetPosition()
        print ac.GetFocalPoint()
        print ac.GetViewUp()
        ac.SetPosition(0.82, -17.12019729614258, 481.4783652957321)
        ac.SetFocalPoint(0.82, -17.12019729614258, 0.)
        ac.SetViewUp(0., 1., 0.)
        show_m.render()
    elif 'FixView' in req_event:
        req_event = 'LeftButtonReleaseEvent' + last_msg + '\n'
        req_event += 'EndInteractionEvent' + last_msg + '\n'
        req_event += 'RenderEven' + last_msg + '\n'
        show_m.play_events(req_event)
        show_m.render()
    else:
        print 'REQ', req_event
        last_msg = req_event.split('\n')[-2]
        last_msg = re.sub("[A-Z,a-z]*", '', last_msg)
        show_m.play_events(req_event)
        show_m.render()
        print 'LAST_', req_event[-1]
        ac = show_m.ren.GetActiveCamera()
        print ac.GetPosition()
        print ac.GetFocalPoint()
        print ac.GetViewUp()
    return 'All done'

run(host='localhost', port=port, debug=True)
#thread = Thread(target=run, kwargs={'host':'localhost', 'port':8080, 'debug':True})
#thread.start()

show_m.start()
