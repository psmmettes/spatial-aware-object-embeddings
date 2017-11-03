#
# Score the boxes based on person detection and interaction with dominant
# objects.
#
# This is step 2 towards zero-shot action localization with spatial-aware
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

import _init_paths
from fast_rcnn.nms_wrapper import nms

import helper
import word_to_vec

from ConfigParser import SafeConfigParser

#
# Parse all arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Detect local objects for a complete action dataset")
    parser.add_argument("-c", dest="configfile", help="Configuration file", default="config/ucf-sports.config", type=str)
    parser.add_argument("-t", dest="topn", help="Number of objects to used per action", default=1, type=int)
    parser.add_argument("-d", dest="bdist", help="Maximum allowed distance between actors and objects (border distance in pixels)", default=25, type=int)
    parser.add_argument("-n", dest="nmst", help="Threshold for non-maximum suppression to score less boxes", default=0.5, type=int)
    parser.add_argument("-w", dest="ow", help="Relative weight of the object score compared to actor scores", default=1.0, type=float)
    args = parser.parse_args()
    return args

#
# Reweight scores based on person scores and relations to objects.
#
def frame_scores(pdata, odata, priors, oweights, ow, bdist):
    scores = pdata[:,-1].copy()
    for i in xrange(pdata.shape[0]):
        for j in xrange(len(odata)):
            bs, bi = -1, -1
            for k in xrange(odata[j].shape[0]):
                if odata[j][k,-1] > bs and helper.box_dist(pdata[i,:4], \
                        odata[j][k,:4]) <= bdist:
                    bs, bi = odata[j][k,-1], k
            distribution = helper.tiledist(odata[j][bi,:4], pdata[i,:4])
            scores[i] += ow * bs * oweights[j] * (1 - \
                    helper.jensen_shannon_divergence(priors[j], distribution))
    return scores

#
# Main object for scoring boxes in video frames based on actors, objects,
# and spatial relations between them.
#
class ScoreBoxes(object):
    
    #
    # Initialize the class by reading the configfile, yielding the videos,
    # and loading the word2vec data + spatial relations.
    #
    def __init__(self, configfile):
        # Parse the configurationfile.
        parser = SafeConfigParser()
        parser.read(configfile)
        self.dataset = configfile.split("/")[1].split(".")[0]
        
        # Yield the appropriate directories.
        self.domdir   = parser.get('actions', 'domdir')
        self.scoredir = parser.get('actions', 'scoredir')
        self.wtvfile  = parser.get('actions', 'wtvfile')
        self.split    = parser.get('actions', 'testsplit')
        self.vidlist  = parser.get('actions', 'vidfile')
        
        # Load the split indices and set the training videos.
        self.teidxs = np.loadtxt(self.split, dtype=int) - 1
        self.videos = [line.strip().split()[0] for line in open(self.vidlist)]
        self.videos = np.array(self.videos)
        self.videos = self.videos[self.teidxs]
        
        # Yield action names.
        self.actions = np.array([vid.split("/")[0] for vid in self.videos])
        self.actions, self.aidxs = np.unique(self.actions, return_inverse=True)
        
        # Load the wordtovec file for the actions.
        self.wtva = np.load(self.wtvfile)
    
        # Actor indices (box coordinates and scores).
        self.pidxs = np.array([85,86,87,88,1])
        
        # Load the prior spatial distributions from file.
        self.priors = np.load("data/mscoco/mscoco_priors_new.npy")
        self.priors = self.priors.reshape(-1)[0]
    
    #
    # For each action in the dataset, find the top n most relevant objects.
    # Relevancy is determined through word2vec.
    #
    def select_local_objects(self, topn):
        # Store for logging purposes.
        self.topn = topn
        
        # MS-COCO class names and vectors.
        self.objectnames = open("data/mscoco/mscoco_classes.txt").readlines()
        self.objectnames = np.array([on.strip() for on in self.objectnames])
        wtvo = np.load("data/mscoco/mscoco-wtv.npy")
        
        # Find the top dominant objects per action.
        self.action_to_object_i = []
        self.action_to_object_w = []
        for i in xrange(len(self.actions)):
            ai, aw = word_to_vec.topn(self.wtva[i], wtvo, topn)
            self.action_to_object_i.append(ai+1)
            self.action_to_object_w.append(aw)
    
    #
    # Perform the actual scoring in here.
    # Score each box in each video frame.
    #
    def score(self, bdist, nmst, ow):
        # Do the scoring for each action for each video.
        for i in xrange(len(self.actions)):
            print "Action %d/%d: %s" %(i+1, len(self.actions), self.actions[i])
            
            # Yield the object indices for the score and box extraction.
            oidxs = []
            for j in xrange(len(self.action_to_object_i[i])):
                oidxs.append(list(range(self.action_to_object_i[i][j]*4+81, \
                        (self.action_to_object_i[i][j]+1)*4+81)) + \
                        [self.action_to_object_i[i][j]])
            
            # Yield the corresponding priors (spatial relations).
            opriors = []
            for j in xrange(len(self.action_to_object_i[i])):
                opriors.append(self.priors[self.objectnames[self.action_to_object_i[i][j]-1]])
            
            # Go over all videos.
            for j in xrange(len(self.videos)):
                print "\tVideo %d/%d\r" %(j+1, len(self.videos)),
                sys.stdout.flush()
                print
                
                # Generate result directory.
                rdir = self.scoredir + "topn-%d/ow-%.2f_bdist-%d_nms-%.3f/%s/%s/" \
                        %(self.topn, ow, bdist, nmst, self.actions[i], self.videos[j])
                if not os.path.exists(rdir):
                        os.makedirs(rdir)

                # Yield frames.
                if self.dataset == "ucf-sports":
                    ddir = self.domdir + self.videos[j]
                elif self.dataset == "j-hmdb":
                    ddir = self.domdir + self.videos[j] + ".avi"
                elif self.dataset == "h2t":
                    ddir = self.domdir + self.videos[j].split("/")[1]
                elif self.dataset == "ucf-101":
                    ddir = self.domdir + self.videos[j].split("/")[1][:-4]
                frames = sorted(os.listdir(ddir))
                
                # Go over all frames.
                for k in xrange(len(frames)):
                    # Skip done work.
                    if os.path.exists(rdir + "frame-%05d-data.npz" %(k)):
                        continue

                    # Load the boxes and scores.
                    framedata  = np.load(ddir + "/" + frames[k])
                    
                    # Person boxes and scores.
                    persondata = framedata[:,self.pidxs]
                    # Reduce computational effort with non-maximum suppression.    
                    pkeep      = nms(persondata, nmst)
                    persondata = persondata[pkeep,:]

                    # Compute scores.
                    if self.topn == 0:
                        # Only use actor (baseline in the paper).
                        newscores = persondata[:,-1].copy()
                    else:
                        # Object boxes and scores.
                        objectdata = []
                        for l in xrange(len(oidxs)):
                            od    = framedata[:,oidxs[l]]
                            okeep = nms(od, nmst)
                            od    = od[okeep,:]
                            objectdata.append(od)
                        
                        # Score per box.
                        newscores = frame_scores(persondata, objectdata, \
                                opriors, self.action_to_object_w[i], ow, bdist)

                    # Store the indices, boxes, and scores.
                    np.savez(rdir + "frame-%05d-data.npz" %(k), boxes=persondata[:,:4], \
                            boxidxs=pkeep, scores=newscores)
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
    topn      = args.topn
    bdist     = args.bdist
    nmst      = args.nmst
    ow        = args.ow
    
    # Initialize and run the box scoring.
    boxmodel = ScoreBoxes(configfile)
    boxmodel.select_local_objects(topn)
    boxmodel.score(bdist, nmst, ow)
