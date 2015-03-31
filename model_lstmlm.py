#!/usr/bin/env python

"""
The implementation of the NNLM model with only local context.
"""

from model_util import *
import io_read
import codecs
import theano
import theano.tensor as T
import numpy as np


class ModelLSTMLM(object):
    def __init__(self, vocab_size, emb_dim, act_func, n_out, pretrain_file):
        
        self.in_vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.activation = get_activation_func(act_func)
        self.n_out = n_out
        self.pretrain_file = pretrain_file
        
    def buildModel(self):
        ##############################
        # MODEL ARCHITECTURE

        # symbolic variables
        self.x = T.matrix('x')
        #self.mask = T.matrix('mask', dtype='float32')
        self.classifier = LSTMLM(self.x, self.in_vocab_size, self.emb_dim, self.n_out, pretrain_file)
        self._initSGD()


    # CALCULATE GRAD USING SYMBOLIC FUNCTION
    def _initSGD(self):
        # symbolic variables
        y = T.ivector('y')
        lr = T.scalar('lr')

        # set symbolic cost function
        cost = self.classifier.nll(y)

        # set symbolic param gradients and adjust learning rate accordingly
        gparams = []
        grad_norm = 0.0
        for param in self.classifier.params:
            gparam = T.grad(cost, param)
            grad_norm += (gparam ** 2).sum()
            # gradients
            gparams.append(gparam)
        grad_norm = T.sqrt(grad_norm)
        max_grad_norm = 5
        if T.gt(grad_norm, max_grad_norm):
            lr = lr * max_grad_norm / grad_norm

        # set symbolic update rules
        updates = []
        for param, gparam in zip(self.classifier.params, gparams):
            updates.append((param, param - lr * gparam))
        
        self.y = y
        self.lr = lr
        self.cost = cost
        self.grad_norm = grad_norm
        self.updates = updates

    #################################################
    # used in train_util #####       BEGIN       ####
    #################################################
    def getTrainModel(self, data_x, data_y):
        self.start_index = T.lscalar()
        self.end_index = T.lscalar()
        self.learning_rate = T.scalar()        

        # TRAIN_MODEL
        self.train_outputs = [self.cost, self.grad_norm]
        self.train_set_x, self.train_set_y = io_vocab.shared_dataset([data_x, data_y])
        self.int_train_set_y = T.cast(self.train_set_y, 'int32')
        self.train_model = theano.function(inputs=[self.start_index, self.end_index, self.learning_rate], outputs=self.train_outputs, updates=self.updates,
                                      givens={
            self.x: self.train_set_x[self.start_index:self.end_index],
            self.y: self.int_train_set_y[self.start_index:self.end_index],
            self.lr: self.learning_rate})

        return self.train_model

    # def getTrainTestModel(self):
        
        
    def getValidationModel(self, valid_set_x, valid_set_y, batch_size):
        self.num_valid_ngrams = valid_set_x.get_value(borrow=True).shape[0]
        self.num_valid_batches = (self.num_valid_ngrams - 1) / batch_size + 1
        self.valid_model = theano.function(inputs=[self.start_index, self.end_index], outputs=self.classifier.sum_ll(self.y),
                                      givens={
            self.x: valid_set_x[self.start_index:self.end_index],
            self.y: valid_set_y[self.start_index:self.end_index],
            self.lr: self.learning_rate})

        return self.valid_model

    def getTestModel(self, test_set_x, test_set_y, batch_size):
        self.num_test_ngrams = test_set_x.get_value(borrow=True).shape[0]
        self.num_test_batches = (self.num_test_ngrams - 1) / batch_size + 1
        self.test_model = theano.function(inputs=[self.start_index, self.end_index], outputs=self.classifier.sum_ll(self.y),
                                      givens={
            self.x: test_set_x[self.start_index:self.end_index],
            self.y: test_set_y[self.start_index:self.end_index],
            self.lr: self.learning_rate})

        return self.test_model

    def updateTrainModelInput(self, data_x, data_y):
        if len(data_y) == 0:
            return False # EOF
        self.train_set_x.set_value(
            np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
        self.train_set_y.set_value(
            np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
        self.int_train_set_y = T.cast(self.train_set_y, 'int32')
        return True

    # used in train_util: num_train_samples = model.getTrainSetXSize()
    def getTrainSetXSize(self):
        return self.train_set_x.get_value(borrow=True).shape[0]

    # for old interface
    def getModelSymbols(self):
        return (self.classifier, self.x, self.y, self.lr, self.cost, self.grad_norm, self.updates)
    #################################################
    # used in train_util #####       END         ####
    #################################################


class LSTMLM(object):

    """
    The complete LSTMLM model.

  
    """

    def __init__(self, input, in_vocab_size, emb_dim, n_out, pretrain_file):
        
        
        self.emb_dim = emb_dim
        self.in_vocab_size = in_vocab_size
        #self.mask=mask
        
        # linear layer
        self.linearLayer = LinearLayer(input, self.in_vocab_size, emb_dim, pretrain_file)
  
        # lstm layer

        prev_layer=self.linearLayer
        
        lstm_layer = LSTMLayer(prev_layer.output, self.emb_dim)
        
        hidden_params=lstm_layer.params
        hidden_out=4*self.emb_dim
        
        

        # softmax layer
        self.softmaxLayer = SoftmaxLayer(self.lstm_layer.output, hidden_out, n_out)

        # nll
        self.nll = self.softmaxLayer.nll

        # sum_ll
        self.sum_ll = self.softmaxLayer.sum_ll

        # params
        self.params = self.linearLayer.params + \
            hidden_params + self.softmaxLayer.params
