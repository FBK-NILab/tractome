import sys
import numpy as np
import nibabel as nib
from dipy.viz import actor, window, ui
from dipy.tracking.streamline import transform_streamlines
from bottle import run, post, request, response
from threading import Thread

global stream_actor, slave_actor, show_m

port = sys.argv[1]
some = 10000
src = '/Users/paolo/Downloads/562345_D10RETEST_b2k_csddet_sift2_FS2016_ioff.left.trk'
src = '/Users/paolo/Datasets/HCP/Demo_Correspondence/sub-627549_size-100k_tract.trk'
t = nib.streamlines.load(src)
tract = transform_streamlines(t.streamlines, np.linalg.inv(t.affine))
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

show_m = window.ShowManager(renderer, size=(600, 600))
show_m.initialize()
show_m.render()
ac = renderer.GetActiveCamera()
print ac.GetPosition()
print ac.GetFocalPoint()
print ac.GetViewUp()

@post('/process')
def my_process():
    global slave_actor, stream_actor
    req_event = request.body.read()
    print req_event
    if 'Expand' in req_event:
        expand = [int(s) for s in req_event.strip('Expand ').split()]
        #expand = [0, 1, 2, 3, 4, 5,]
        subset = np.take(tract, expand)
        slave_actor = actor.line(subset)
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
        ac.SetPosition(71.42986869812012, 87.12019729614258, 443.4783652957321)
        ac.SetFocalPoint(71.42986869812012, 87.12019729614258, 57.54511070251465)
        ac.SetViewUp(0., 1.00, 0.)
        show_m.render()
    else:
        print req_event
        show_m.play_events(req_event)
        show_m.render()
    return 'All done'

run(host='localhost', port=port, debug=True)
#thread = Thread(target=run, kwargs={'host':'localhost', 'port':8080, 'debug':True})
#thread.start()

show_m.start()
