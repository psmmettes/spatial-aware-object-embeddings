#
# Extract the boxes and scores for the objects from a Faster R-CNN model,
# applied to a complete dataset.
#
# This is step 1 towards zero-shot action localization with spatial-aware
# object embeddings.
#
# Pascal Mettes (2017).
#
# Please cite accordingly when using this code:
#
# @article{mettes2017spatial,
#  title={Spatial-Aware Object Embeddings for Zero-Shot Localization and
#  Classification of Actions},
#  author={Mettes, Pascal and Snoek, Cees G M},
#  journal={ICCV},
#  year={2017}
#}
#

import os
import sys
import cv2
import h5py
import numpy as np
import scipy.io as sio
import argparse
from   ConfigParser import SafeConfigParser

import caffe
import _init_paths
from   fast_rcnn.config import cfg
from   fast_rcnn.test import im_detect
from   fast_rcnn.nms_wrapper import nms
from   utils.timer import Timer

#
# Parse all arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Detect local objects for a complete action dataset")
    parser.add_argument("-c", dest="configfile", help="Configuration file", default="config/ucf-sports.config", type=str)
    parser.add_argument("-s", dest="stride", help="Frame stride for object detection", default=1, type=int)
    parser.add_argument("-p", dest="prototxt", help="Location for Caffe prototxt (Faster R-CNN)", default="", type=str)
    parser.add_argument("-m", dest="caffemodel", help="Location for Caffe model (Faster R-CNN)", default="", type=str)
    args = parser.parse_args()
    return args

#
# Main entry point fo the script. In here: go through all videos and frames.
#
if __name__ == "__main__":
    # User parameters.
    args       = parse_args()
    config     = args.configfile
    stride     = args.stride
    prototxt   = args.prototxt
    caffemodel = args.caffemodel
    
    # Yield directories and videos.
    parser   = SafeConfigParser()
    parser.read(config)
    framedir = parser.get('actions', 'framedir')
    resdir   = parser.get('actions', 'scoredir')
    vidlist  = parser.get('actions', 'vidfile')
    videos   = np.array([v.strip().split()[0] for v in open(vidlist)])

    # Caffe / Faster R-CNN settings.
    ### Use region proposals.
    cfg.TEST.HAS_RPN = True
    ### Set to CPU/GPU.
    caffe.set_mode_gpu()
    
    # Load the network.
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print "HERE"

    # Go over all videos and all frames.
    for video in videos:
        print video
        frames = sorted(os.listdir(framedir + video))

        # Create resulting directory if not existing already.
        if not os.path.exists(resdir + video):
            os.makedirs(resdir + video)

        # Go over the frames.
        for i in xrange(0,len(frames),stride):
            print i,
            sys.stdout.flush()

            # Skip if done before.
            if os.path.exists(resdir + video + "/" + frames[i][:-3] + "npy"):
                continue

            # Load frame, get data, stack data, store data.
            try:    
                frame = cv2.imread(basedir + video + "/" + frames[i])
                s,b   = im_detect(net, frame)
                t     = np.hstack((s,b))
                np.save(resdir + video + "/" + frames[i][:-3] + "npy", t)
            except:
                pass
    print
