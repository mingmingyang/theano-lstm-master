#!/usr/bin/env python
"""
A module that is used to map a io_vocab format data file into a integer file. The entire workflow consists of:
  1. First scan the file to extract vocabulary, according to a given vocab size; or read vocabulary from file if it already exists.
  2. Write the vocabulary to file if vocabulary not exists.
  3. Scan the file again: readin a sentence line, convert it into integer sequences, and write to output file.
Example input:
  - filename = training
  - vocab_size = 1000
  - vocab_file_prefix

Example output:
  - training.1000.id.sentences
  - training.1000.vocab
  - training.1000.id.words

If the vocab_file exists, then it should use that file instead of creating new vocabs.

"""

usage = "A module that is used to map a io_vocab format data file into a integer file."

import os
import sys
import time
import re
import codecs
import argparse
import datetime

# our libs
import io_vocab

def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """	
  parser = argparse.ArgumentParser(description=usage) # add description
  # positional arguments
  parser.add_argument('filename', metavar='filename', type=str, help='the input file name (without extension)') 
  parser.add_argument('vocab_size', metavar='vocab_size', type=int, help='the expected vocab size') 
  parser.add_argument('vocab_file_prefix', metavar='vocab_file_prefix', type=str, help='the prefix of vocab file') 

  args = parser.parse_args()
  return args


def map_file(filename, vocab_size, vocab_file_prefix):
  """
  Extract vocab, and convert file into integer sequences, and write to output file.
  """	
  input_file = filename
  vocab_file = vocab_file_prefix + '.' + str(vocab_size) + '.vocab'
  (vocab_map, v) = io_vocab.get_vocab(input_file, vocab_file, -1, vocab_size)
  print "# vocab size is %d" % v
  output_sentences_file = filename + '.' + str(vocab_size) + '.id.sentences'
  sentences = io_vocab.get_mapped_sentence(input_file, vocab_map, output_sentences_file)
  max_len = max([len(x) for x in sentences])
  print "# max length of sentence is %d" % max_len  
  output_words_file = filename + '.' + str(vocab_size) + '.id.words'
  words = io_vocab.get_mapped_words(sentences,output_words_file)
  
def main():
	args = process_command_line()
	map_file(args.filename, args.vocab_size, args.vocab_file_prefix)

if __name__ == '__main__':
	main()
