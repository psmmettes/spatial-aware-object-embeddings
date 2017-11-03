#
# Use the box scores of individual frames to generate action tubes.
#
# This is step 3 towards zero-shot action localization with spatial-aware
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
import h5py
import numpy as np
import argparse

import helper

from ConfigParser import SafeConfigParser

#
# Parse all arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Detect local objects for a complete action dataset")
    parser.add_argument("-c", dest="configfile", help="Configuration file", default="config/ucf-sports.config", type=str)
    parser.add_argument("-b", dest="basedir", help="Directory where data from step 2 is stored", default="", type=str)
    parser.add_argument("-i", dest="iou_weight", help="Weight for the overlap score when linking boxes", default=1.0, type=float)
    parser.add_argument("-s", dest="stride", help="Stride between frames when computing scores (for interpolation)", default=1, type=int)
    parser.add_argument("-t", dest="nr_tubes", help="Number of tubes per action per video", default=5, type=int)
    args = parser.parse_args()
    return args

#
# Generate tubes by linking high scoring spatial-aware object embedding boxes.
#
class GenerateTubes(object):
    
    #
    # Initialize the class by settng the paths and loading the videos.
    #
    def __init__(self, configfile, basedir):
        # Parse the configurationfile.
        parser = SafeConfigParser()
        parser.read(configfile)
        self.dataset = configfile.split("/")[1].split(".")[0]
        
        # Yield the appropriate directories.
        self.gtdir    = parser.get('actions', 'gtdir')
        self.domdir   = parser.get('actions', 'domdir')
        self.scoredir = parser.get('actions', 'scoredir')
        self.tubedir  = parser.get('actions', 'tubedir')
        self.framedir = parser.get('actions', 'framedir')
        self.wtvfile  = parser.get('actions', 'wtvfile')
        self.split    = parser.get('actions', 'testsplit')
        self.vidlist  = parser.get('actions', 'vidfile')
        
        self.basedir  = basedir
        
        # Load the split indices and set the training videos.
        self.teidxs = np.loadtxt(self.split, dtype=int) - 1
        self.videos = [line.strip().split()[0] for line in open(self.vidlist)]
        self.videos = np.array(self.videos)
        self.videos = self.videos[self.teidxs]
 
        # Yield action names.
        self.actions = np.array([vid.split("/")[0] for vid in self.videos])
        self.actions, self.aidxs = np.unique(self.actions, return_inverse=True)
   
    #
    # Generate the tubes from scores.
    #
    def generate(self, nr_tubes, iou_weight, stride):
        # Do the scoring for each action for each video.
        for i in xrange(len(self.actions)):
            print "Action %d/%d: %s" %(i+1, len(self.actions), self.actions[i])
            
            # Go through all videos.
            for j in xrange(len(self.videos)):
                print "\tVideo %d/%d\r" %(j+1, len(self.videos)),
                sys.stdout.flush()
                
                # Initialize the path wherr the tubes will be stored.
                bsplit = self.basedir.split("/")
                h5d = self.tubedir + "%s/%s/nrtubes-%d_iouw-%.2f/%s/%s/" \
                        %(bsplit[-3], bsplit[-2], nr_tubes, iou_weight, \
                        self.actions[i], self.videos[j])
                h5f = h5d + "/tubes.hdf5"
                if os.path.exists(h5f):
                    continue
                if not os.path.exists(h5d):
                    os.makedirs(h5d)
                
                # Yield frames.
                bdir   = self.basedir + self.actions[i] + "/" + self.videos[j]
                frames = sorted(os.listdir(bdir))
                framescores = []
                frameboxes  = []
                print bdir
                
                # Number of frames (for interpolation).
                if self.dataset == "h2t":
                    flist = os.listdir(self.framedir + self.videos[j].split("/")[1])
                    nrframes = len(flist)
                elif self.dataset == "ucf-101":
                    flist = os.listdir(self.framedir + self.videos[j].split("/")[1][:-4])
                    nrframes = len(flist)
                else:
                    nrframes = len(frames)
                
                # Go over all frames.
                for k in xrange(len(frames)):
                    framedata = np.load(bdir + "/" + frames[k])
                    framescores.append(framedata['scores'])
                    frameboxes.append(framedata['boxes'])
                
                # Open the file where to save the tubes.
                h5f = h5py.File(h5f, "w")
                
                # Generate the tubes.
                for k in xrange(nr_tubes):
                    ds, di = [], []
                    print k, nr_tubes
                    
                    # Initialize.
                    for l in xrange(len(frameboxes)):
                        frsize = frameboxes[l].shape[0]
                        ds.append(np.zeros(frsize))
                        di.append(np.zeros(frsize, dtype=int))
                    
                    # Viterbi.
                    for l in xrange(len(frameboxes)-2, -1, -1):
                        cscores  = helper.viterbi_scores(frameboxes[l], \
                                framescores[l], frameboxes[l+1], \
                                framescores[l+1], iou_weight)
                        cscores += ds[l+1]
                        ds[l] = np.max(cscores, 1)
                        di[l] = np.argmax(cscores, 1)
                    
                    # Start from frame 0.
                    sf, si = 0, np.argmax(ds[0])
                    tube   = [frameboxes[sf][si]]
                    tscore = [framescores[sf][si]]
                    tubeidxs = [sf*stride]
                    idx    = si
                    framescores[sf][si] = 0
                    
                    # Connect.
                    for l in xrange(sf, len(frameboxes)-1):
                        idx = di[l][idx]
                        tube.append(frameboxes[l+1][idx])
                        tscore.append(framescores[l+1][idx])
                        tubeidxs.append((l+1)*stride)
                        framescores[l+1][idx] = 0
                    
                    # Numpy format and save.
                    tube     = np.vstack(tube)
                    tscore   = np.array(tscore)
                    tubeidxs = np.array(tubeidxs)
                    tube     = np.hstack((tubeidxs[:,np.newaxis], tube))
                    
                    # Interpolation.
                    if stride > 1:
                        tube, tscore = helper.tube_interpolate(tube, tscore, stride, nrframes)

                    d1   = h5f.create_dataset(str(k), tube.shape)
                    d1[:,:] = tube.copy()
                    d2   = h5f.create_dataset("scores-%d" %(k), tscore.shape)
                    d2[:] = tscore.copy()
                
                # Close the file.
                h5f.close()
            print

#
# Entry point of the script.
#
if __name__ == "__main__":
    # User parameters.
    args = parse_args()

    # Specify which dataset to use.
    configfile = args.configfile
    
    # Hyperparameters.
    nr_tubes   = args.nr_tubes
    basedir    = args.basedir
    iou_weight = args.iou_weight
    stride     = args.stride
     
    tg = GenerateTubes(args.configfile, args.basedir)
    tg.generate(args.nr_tubes, args.iou_weight, args.stride)
