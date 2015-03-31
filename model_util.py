#!/usr/bin/env python

"""
The definition (modelling) and initialization of shared layers in the neural language model architecture.
"""

debug = False

import sys
import re
import codecs

import cPickle
import random
import numpy as np
import scipy.sparse as sp
import theano
import theano.tensor as T
import theano.sparse as S
reload(sys)
sys.setdefaultencoding("utf-8")

import io_vocab


rng = np.random.RandomState(1234)
init_range = 0.05



def sigmoid(x):
    return T.nnet.sigmoid

def get_activation_func(act_func):
    if act_func == 'tanh':
        # sys.stderr.write('# act_func=tanh\n')
        activation = T.tanh
    elif act_func == 'relu':
        # sys.stderr.write('# act_func=rectifier\n')
        activation = rectifier
    elif act_func == 'leakyrelu':
        # sys.stderr.write('# act_func=leaky rectifier\n')
        activation = leaky_rect
    elif act_funct == 'sigmoid':
        activation = sigmoid
    else:
        sys.stderr.write(
            '! Unknown activation function %s, not tanh or relu\n' % (act_func))
        sys.exit(1)
    return activation


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

class SoftmaxLayer(object):

    """
    class SOFTMAXLAYER

    The class takes the output from the last hidden layer as input and compute the log probability of
    each possible next word in the vocab with a softmax function. Numerical stability is ensured.

    Argument:
      - input: a matrix where each row is output from last hidden layer, and row number equals to batch size

    Adapt from this tutorial http://deeplearning.net/tutorial/logreg.html
    """

    def __init__(self, input, in_size, out_size, softmax_W=None, softmax_b=None):
        global rng
        global init_range
        if softmax_W is None or softmax_b is None:
            # randomly intialize everything
            softmax_W = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(in_size, out_size)), dtype=theano.config.floatX)
            softmax_b = np.zeros((out_size,), dtype=theano.config.floatX)
        else:
            given_in_size, given_out_size = softmax_W.shape
            assert(given_in_size == in_size and given_out_size == out_size)

        # shared variables
        self.W = theano.shared(value=softmax_W, name='W', borrow=True)
        self.b = theano.shared(value=softmax_b, name='b', borrow=True)

        # compute the "z = theta * x" in traditional denotation
        # input: batch_size * hidden_dim
        # self.W: hidden_dim * |V|
        # self.b: 1 * |V|
        x = T.dot(input, self.W) + self.b

        # take max for numerical stability
        x_max = T.max(x, axis=1, keepdims=True)

        # Take the log of the denominator in the softmax function
        # 1. minus max from the original x so that numerical stability can be ensured when we take exp
        # 2. we can add the max back directly since we take log after
        # the exp
        self.log_norm = T.log(
            T.sum(T.exp(x - x_max), axis=1, keepdims=True)) + x_max

        # The log probability equals to the log numerator (theta * x)
        # minus the log of the denominator (log norm)
        self.log_p_y_given_x = x - self.log_norm

        # params
        self.params = [self.W, self.b]

    def nll(self, y):
        """
        Mean negative log-lilelihood
        """
        return -T.mean(self.log_p_y_given_x[T.arange(y.shape[0]), y])

    def sum_ll(self, y):
        """
        Sum log-lilelihood
        """
        return T.sum(self.log_p_y_given_x[T.arange(y.shape[0]), y])

    def ind_ll(self, y):
        """
        Individual log-lilelihood
        """
        return self.log_p_y_given_x[T.arange(y.shape[0]), y]



def load_pretrain_emb(pretrain_file):
    f = file(pretrain_file, 'rb')
    linear_W_emb = cPickle.load(f)
    linear_W_emb = np.float32(linear_W_emb)
    return linear_W_emb
                          
class LinearLayer(object):

    """
    class LINEARLAYER

    The linear layer used to compute the word embedding matrix. This class actually take the input
    and then COMPUTE the output using linear_W_emb and stores the result in self.output.

    Argument:
      - input: a matrix where each row is a word vector, and row number equals to batch size.
    """

    # LINEAR LAYER MATRIX of dim(vocab_size, emb_dim)
    def __init__(self, input, vocab_size, emb_dim, pretrain_file, linear_W_emb=None):

        global rng
        global init_range
        if pretrain_file:
            linear_W_emb = load_pretrain_emb(pretrain_file)
            print "* Using pretrained linear_W_emb ..."
            assert(len(linear_W_emb) == vocab_size)
        else:
            linear_W_emb = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(vocab_size, emb_dim)), dtype=theano.config.floatX)

        # shared variables
        self.W_emb = theano.shared(value=linear_W_emb, name='W_emb')

        # stack vectors
        input = T.cast(input, 'int32')

        # output is a matrix where each row correponds to a embedding vector, and row number equals to batch size
        # output dimensions: batch_size * emb_dim)
        if input.ndim==1:  
            self.output = self.W_emb[input.flatten()].reshape((input.shape[0], emb_dim))
        if input.ndim==2:
            self.output = self.w_emb[input.flatten()].reshape((input.shape[0], input.shape[1]*emb_dim))

        # params is the word embedding matrix
        self.params = [self.W_emb]




####################################
# class HIDDENLAYER
####################################

class LSTMLayer(object):
    def __init__(self, input, out_size):
        """
        class LSTMLAYER

        The class actually takes the input and then COMPUTE the output using W_values, U_values and b_values
        and stores the result in self.output.

        Argument:
          - input: a matrix where each row is the last layer ouput, and the row number equals to batch size
        """

        global rng
        global init_range
        lstm_W = np.concatenate([ortho_weight(out_size),
                           ortho_weight(out_size),
                           ortho_weight(out_size),
                           ortho_weight(out_size)], axis=1)
        
        lstm_U = np.concatenate([ortho_weight(out_size),
                           ortho_weight(out_size),
                           ortho_weight(out_size),
                           ortho_weight(out_size)], axis=1)
        
        lstm_b = np.zeros((4*out_size,), dtype=theano.config.floatX)

        # mask
        #self.mask=mask
        # W
        self.W = theano.shared(value=lstm_W, name='W')
        # U
        self.U = theano.shared(value=lstm_U, name='U')                  
        # b
        self.b = theano.shared(value=lstm_b, name='b')
        
        # state_below
        self.state_below=input
        # n_samples
        self.n_samples=input.shape[0]

        # emb_dim
        self.emb_dim=out_size
        # h_
        lstm_h_ = tensor.alloc(0., self.n_samples,self.emb_dim)
        self.h_ = theano.shared(value=lstm_h_, name='h_')
        # c_
        lstm_c_ = tensor.alloc(0., self.n_samples,self.emb_dim)
        self.c_ = theano.shared(value=lstm_h_, name='c_')
        # output
        self.output = output()

        # params
        self.params = [self.W, self.U, self.b, self.h_, self.c_]
                          
   
        
                          
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(x_, h_, c_):
        preact = tensor.dot(h_, self.U)
        preact += x_
        preact += self.b

        i = tensor.nnet.sigmoid(_slice(preact, 0, self.emb_dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, self.emb_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, self.emb_dim))
        c = tensor.tanh(_slice(preact, 3, self.emb_dim))

        c = f * c_ + i * c
        #c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        #h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    def output():
        state_below = (tensor.dot(self.state_below, self.W) +
                   self.b)

    
        rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[self.h_, self.c_],
                                name='lstm_layer')
        
        return T.dot(rval[0],self.U)+self.b





