#!/usr/bin/env python

"""
Train a LSTMLM model.
"""

usage = 'To train LSTMLM using Theano'

import cPickle
import gzip
import os
import sys
import time
import re
import codecs
import argparse
import datetime
import copy

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict

# our libs
import model_lstmlm
import io_vocab
import io_read
import io_model
from train_util import *

def process_command_line():
    """
    Return a 1-tuple: (args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """

    parser = argparse.ArgumentParser(description=usage)  # add description
    # positional arguments
    parser.add_argument(
        'train_file', metavar='train_file', type=str, help='train file')
    parser.add_argument(
        'valid_file', metavar='valid_file', type=str, help='valid file')
    parser.add_argument(
        'test_file', metavar='test_file', type=str, help='test file')
    parser.add_argument(
        'vocab_size', metavar='vocab_size', type=int, help='vocab size')
    parser.add_argument(
        'vocab_file', metavar='vocab_file', type=str, help='vocab file')

    # optional arguments
    parser.add_argument('--emb_dim', dest='emb_dim', type=int,
                        default=128, help='embedding dimension (default=128)')
    parser.add_argument('--vocab_size', dest='vocab_size', type=int,
                        default=10000, help='number of words (default=10000)')                                  
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=float, default=0.0001, help='learning rate (default=0.0001)')
    parser.add_argument('--chunk', dest='chunk', type=int, default=2000,
                        help='each time consider batch_size*chunk  (default=2000)')
    #parser.add_argument('--batch_size', dest='batch_size', type=int, default=16,
    #                    help='each time consider batch_size (default=16)')                                        
    #parser.add_argument('--valid_batch_size', dest='valid_batch_size', type=int, default=16,
    #                    help='each time consider valid_batch_size (default=16)')
    parser.add_argument('--valid_freq', dest='valid_freq',
                        type=int, default=10000, help='valid freq (default=10000)')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='sgd',
                        help='optimizer: sgd -- use SGD, adadelta -- use adadelta')
    parser.add_argument('--act_func', dest='act_func', type=str, default='tanh',
                        help='non-linear function: \'tanh\' or \'relu\' (default=\'tanh\')')
    parser.add_argument('--finetune', dest='finetune', type=int, default=1,
                        help='after training for this number of epoches, start halving learning rate(default: 1)')

    parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=5,
                        help='number of epochs, i.e. how many times to go throught the training data (default: 5)')
    

    # pretraining file
    parser.add_argument('--pretrain_file', dest='pretrain_file', type=str,
                        default=None, help='pretrain file for linear_W_emb layer (default=None)')
    parser.add_argument('--save_model', dest="save_model_file", type=str, default=None, help="save the model parameters to file")

    args = parser.parse_args()
    return args
    
class TrainLSTMModel(TrainModel):
    def loadModelParams(self, model):
        self.model = model
        self.model_param_loaded = True

    #def loadTrainSet(self, train_data_package):
        
    def getBatchData(self):
     

        is_shuffle = self.is_shuffle
        chunk_size = self.chunk_size
        opt=self.opt

        (data_x, data_y) = io_read.get_all_data(self.train_f, num_read_lines=chunk_size)
        return (data_x, data_y)   

       

    def loadValidSet(self, valid_data_package):
        self.valid_set_x, self.valid_set_y = valid_data_package
        self.shared_valid_set_x, self.shared_valid_set_y = io_read.shared_dataset(valid_data_package)
        self.shared_valid_set_y = T.cast(self.shared_valid_set_y, 'int32')
        self.valid_set_loaded = True

    def loadTestSet(self, test_data_package):
        self.test_set_x, self.test_set_y = test_data_package
        self.shared_test_set_x, self.shared_test_set_y = io_read.shared_dataset(test_data_package)
        self.shared_test_set_y = T.cast(self.shared_test_set_y, 'int32')
        self.test_set_loaded = True


    def loadBatchData(self, isInitialLoad=False):
  
        vocab_size = self.vocab_size
        

        chunk_size = self.chunk_size
        
        

        (self.data_x, self.data_y) = io_read.get_all_data(train_f, num_read_lines=chunk_size)

        if isInitialLoad == False:
            assert(type(self.model)==model_lstmlm.ModelLSTMLM)
            return self.model.updateTrainModelInput(self.data_x, self.data_y)

    def trainOnBatch(self, train_model, i, batch_size, num_train_batches, num_train_samples, learning_rate):
        ngram_start_id = i * batch_size
        ngram_end_id = (i + 1) * batch_size if i < (num_train_batches - 1) else num_train_samples
        outputs = train_model(ngram_start_id, ngram_end_id, learning_rate)
        return outputs

    def buildModels(self):
        assert(hasattr(self, 'model'))
        print "Getting train model ..."
        train_model = self.model.getTrainModel(self.data_x, self.data_y)
        print "Getting validation model ..."
        valid_model = self.model.getValidationModel(self.shared_valid_set_x, self.shared_valid_set_y, self.batch_size)
        print "Getting test model ..."
        test_model = self.model.getTestModel(self.shared_test_set_x, self.shared_test_set_y, self.batch_size)
 
        print "Going to start training now ..."
        return (train_model, valid_model, test_model)

    def validate(self, model, num_ngrams, batch_size, num_batches):
        """
        Return average negative log-likelihood
        """
        loss = 0.0
        for i in xrange(num_batches):
            ngram_start_id = i * batch_size
            ngram_end_id = (i + 1) * batch_size if i < (num_batches - 1) else num_ngrams
            loss -= model(ngram_start_id, ngram_end_id)  # model returns sum log likelihood
        loss /= num_ngrams
        perp = np.exp(loss)
        return (loss, perp)

    def test(self, model, num_ngrams, batch_size, num_batches):
        """
        Return average negative log-likelihood
        """
        loss = 0.0
        for i in xrange(num_batches):
            ngram_start_id = i * batch_size
            ngram_end_id = (i + 1) * batch_size if i < (num_batches - 1) else num_ngrams
            loss -= model(ngram_start_id, ngram_end_id)  # model returns sum log likelihood
        loss /= num_ngrams
        perp = np.exp(loss)
        return (loss, perp)


if __name__=='__main__':
    args=process_command_line()
    print "Process ID: %d"%(os.getpid())
    print_cml_args(args)
    emb_dim=args.emb_dim #128
    train_file=args.train_file
    learning_rate=args.learning_rate
    valid_freq=args.valid_freq
    batch_size=args.batch_size
    vocab_size=args.vocab_size
    optimizer=args.optimizer
    chunk_size=args.chunk
    act_func = args.act_func
    n_epochs=args.n_epochs
    pretrain_file=args.pretrain_file
		####################################
    # LOAD VACAB
    # <words> is a list of words as in string
    # <vocab_map> is a dict mapping from word string to integer number of 1,2,...|Vocab|
    # <vocab_size> is the size of vocab == len(words) == len(vocab_map).
    

    vocab_file=args.vocab_file+'.'+str(args.vocab_size)+'.vocab'
    (vocab_map, vocab_size) = io_vocab.load_vocab(vocab_file)		
		
		####################################
    # LOAD DATA
    # <train_set_x, train_set_y> train samples 
    # <valid_set_x, valid_set_y> valid samples
    # <test_set_x, test_set_y> test samples
    #train_file=args.train_file+'.'+str(args.vocab_size)+'.id.words'
    #(train_set_x, train_set_y)=io_read.get_all_data(train_file) 
       
    valid_file=args.valid_file+'.'+str(args.vocab_size)+'.id.words'
    (valid_set_x, valid_set_y)=io_read.get_all_data(codecs.open(valid_file, 'r', 'utf-8'))
    
    test_file=args.test_file+'.'+str(args.vocab_size)+'.id.words'
    (test_set_x, test_set_y)=io_read.get_all_data(codecs.open(text_file, 'r', 'utf-8'))

    #####################################
    # BUILD MODEL
    print "Start modeling part..."
    lstmlm_model = model_lstmlm.ModelLSTMLM(vocab_size, emb_dim, act_func, vocab_size, pretrain_file)
    lstmlm_model.buildModel()
    
    ####################################
    # START TRAINING
    train_model = TrainLSTMModel()
    train_model.loadVocab(vocab_size, vocab_file)
    train_model.loadValidSet((valid_set_x, valid_set_y))
    train_model.loadTestSet((test_set_x, test_set_y))
    train_model.loadModelParams(lstmlm_model)
    train_model.loadTrainParams(train_file, batch_size, learning_rate, valid_freq, finetune_epoch, chunk_size, vocab_size, n_epochs)
    print "Start training part... (2/2: training)"
    train_model.train()

    # Finish training, save model to file
    if args.save_model_file is not None:
        io_model.save_model(args.save_model_file, lstmlm_model.classifier)
