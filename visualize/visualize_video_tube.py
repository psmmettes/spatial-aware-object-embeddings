#
# This script visualizes a set of 3D tubes through a video.
#
# Written by:    Pascal Mettes.
# Original code: Jan van Gemert.
#
# 2017
#

import os
import sys
import glob
import h5py
from PIL import Image
import numpy as np
import mayavi
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input

#
# Load a set of figures from a directory.
#
# Input:
# framedir (str) - Directory containing the frames.
# ext (str)      - Frame file extension.
#
# Output:
# 4D tensor (nrf, nry, nrx, nrc).
#
def load_frames(framedir, ext="jpg"):
    # Load the sort the frames.
    frames = os.path.join(framedir, "*." + ext)
    frames = glob.glob(frames)
    frames.sort()
    
    # Load frame info.
    data = []
    for i in xrange(len(frames)):
        frame = np.array(Image.open(frames[i]))
        data.append(frame)
    data = np.array(data)
    
    return data

#
# Convert frame data to an actor for 3D visualization.
#
# Input:
# frame (numpy ndarray) - Frame data.
# width (int)           - Width.
# height (int)          - Height.
# b (int)               - Border.
#
# Output:
# Tvtk actor of the frame.
#
def frame_to_actor(frame, width, height, b=5):
    # Dimensions with borders.
    w, h = width+2*b, height+2*b

    # Initialize array.
    data = np.zeros((w, h, 4))
    if frame is None:
        data[:,:,-1] = 255
        data[b:-b,b:-b,-1] = 0
    else:
        data[b:-b,b:-b,:3] = frame
        data[:,:,-1] = 255
    data = np.transpose(data, (1,0,2))
    colors = np.reshape(data, (w*h,4))
    
    # Create actor.
    image = tvtk.ImageData()
    image.point_data.scalars=tvtk.UnsignedCharArray()
    image.point_data.scalars.from_array(colors)
    image.dimensions = np.array((w, h, 1))
    actor = tvtk.ImageActor()
    configure_input(actor, image)
    
    return actor

#
# Visualize an individual tube through the video.
#
# Input:
# tube (numpy ndarray) - Tube content.
# color (rgb tuple)    - Tube color.
# width int)           - Width.
# height (int)         - Height.
# nr_frames (int)      - Number of frames in whole video.
# extent (float)       - Temporal scale factor.
# b (int)              - Border.
#
# Output:
# -
#
def visualize_tube(tube, color, width, height, nr_frames, extent, b=5):
    # Fill a volume with tube data.
    cube = np.zeros((width, height, nr_frames))
    for i in xrange(tube.shape[0]):
        fidx = int(nr_frames - tube[i,0] - 1)
        box  = tube[i,1:] + b
        box  = box.astype(int)
        xmin, ymin, xmax, ymax = box
        cube[ymin:ymax,xmin:xmax, fidx] = 1.0
   
    # Create scalar field.
    field = mlab.pipeline.scalar_field(cube)
    # Re-adjust temporal scale.
    field.spacing = [1,1,extent]
    field.update_image_data = True

    # Plot the field.
    iso = mlab.pipeline.iso_surface(field, opacity=0.4, color=color)
    iso.scene.anti_aliasing_frames = 1
    iso.contour.number_of_contours = 2

#
# Visualize the 3D scene of a video and a set of colored tubes.
#
# Input:
# frames (nrf, nrx, nry, nrc) - Frame data.
# tubes (list)                - List of tubes.
# colors (list)               - Color per tube.
# extent (float)              - Temporal scale.
# nr_middle_frames (int)      - Number of frames in the middle (border only).
#
# Output:
# -
#
def visualize_scene(frames, tubes, colors, extent, nr_middle_frames=0):
    # Create the figure.
    figure = mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1), size=(1600,900))
    figure.scene.anti_aliasing_frames = 1
    
    # Retrieve frame dimensions.
    [nrf, nrx, nry, nrc] = frames.shape
    
    # Set camera position.
    figure.scene.z_plus_view()
    mayavi.mlab.roll(-90)
    figure.scene.parallel_projection = True
    figure.scene.camera.azimuth(25)
    figure.scene.camera.elevation(30)
    figure.scene.camera.zoom(1.5)
    
    # Show end frame.
    endframe = frame_to_actor(frames[-1], nrx, nry)
    endframe.position = [0,0,0]
    figure.scene.add_actor(endframe)
    
    # Show optionaly intermediate frame positions with only black borders.
    fs = nrf*extent / (nr_middle_frames+1)
    for i in xrange(nr_middle_frames):
        frame = frame_to_actor(None, nrx, nry)
        frame.position = [0,0,fs*(i+1)]
        figure.scene.add_actor(frame)
    
    # Show tubes.
    for i in xrange(len(tubes)):
        visualize_tube(tubes[i], colors[i], nrx, nry, nrf, extent)
    
    # Show start frame.
    startframe = frame_to_actor(frames[0], nrx, nry)
    startframe.position = [0,0,nrf*extent]
    figure.scene.add_actor(startframe)
    
    # Show the result and write it to file.
    mlab.show()

#
# Run an example visualization when running the script directly.
#
if __name__ == "__main__":
    # Directory with frames.
    framedir = "example/frames/"
    # Load a tube.
    tubes    = [np.array(h5py.File("example/tube.hdf5", "r")["0"])]
    # Color of the tube.
    colors   = [(1,0,0)]
    # Temporal extent of the video.
    extent   = 30
    # Number of support frames in the middle.
    nr_middle_frames = 1
    
    # Load the frames.
    frames   = load_frames(framedir)
    
    # Visualize.
    visualize_scene(frames, tubes, colors, extent, nr_middle_frames)
