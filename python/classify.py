#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import re
import argparse
import glob
import time
import numpy
import caffe
import googlenet_class_labels

def load_txt_file(path):
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    return lines

def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('(\-?[0-9]+)', s) ]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l

    return sort_nicely(list_of_file_names)

def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = numpy.split(numpy.asarray(idx), n_minibatches)
    else:
        print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
        minibatch_idx = numpy.split(numpy.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [numpy.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx

def main(argv):
    gts = load_txt_file('/data/lisatmp3/yaoli/caffe/caffe/data/ilsvrc12/val.txt')
    gts = [int((gt.split(' ')[-1]).strip()) for gt in gts]
    
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--input_file",
        default='/data/lisatmp3/yaoli/datasets/ILSVRC2012/ILSVRC2012/valid',
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "--output_file",
        default='image_valid_prediction.npy',
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_googlenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_googlenet/bvlc_googlenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        default=True,
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        default=True,
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='JPEG',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, gpu=args.gpu, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    if args.gpu:
        print 'GPU mode'

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        files = glob.glob(args.input_file + '/*.' + args.ext)
        files = sort_by_numbers_in_file_name(files)
    else:
        inputs = [caffe.io.load_image(args.input_file)]

    # Classify.
    start = time.time()
    idx = generate_minibatch_idx(len(files), 100)
    preds = []
    for i, index in enumerate(idx):
        current = [files[j] for j in index]
        gt = [gts[j] for j in index]
        inputs =[caffe.io.load_image(im_f)
                     for im_f in current]
        probs = classifier.predict(inputs, not args.center_only)
        predictions = probs.argmax(axis=1)
        preds += predictions.tolist()
        print '%d / %d minibatches, current acu %.4f'%(i, len(idx), numpy.mean(predictions == gt))
    #label = [googlenet_class_labels.get_googlenet_class_label(pred) for pred in predictions]
    #label_gt = [googlenet_class_labels.get_googlenet_class_label(pred) for pred in gts]
    print 'overall acu ', numpy.mean(preds == gts) 
    print "Done in %.2f s." % (time.time() - start)
    
    # Save
    np.save(args.output_file, predictions)


if __name__ == '__main__':
    main(sys.argv)
