#!/usr/bin/env python

"""
Util functions that are used to parse data into n-gram inputs.
Note that the data should NOT be the raw text file. The vocabulary should be extracted
and each text sentence should be mapped to a sequence of corresponding indeces in the
vocabulary.
"""

debug = False

import sys
import re
import codecs

import cPickle
import random
import numpy as np
import theano
reload(sys)
sys.setdefaultencoding("utf-8")

import io_vocab

def shared_dataset(data_package):
    shared_package = ()
    for d in data_package:
        shared_package += (theano.shared(
        np.asarray(d, dtype=theano.config.floatX), borrow=True), )
    return shared_package


    
def get_all_data(input_f, num_read_lines=-1):
    
    

    x = []  # training examples
    y = []  # labels 
    num_lines=0
    while 1:
        line=input_f.readline().strip()
        if not line:
            break
        tokens=re.split('\s+',line.strip())
        x.append(tokens[0])
        y.append(tokens[1])
        num_lines += 1
        if num_lines == num_read_lines:
            break
    return (x, y)
        
