# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

This program attempts to implement the previous examples in the context of document recognition
Option 1. Word lists with numeric variables are put it for testing if it is the target label

Option 2. Images are fed to the VGG net for deciding the target coordinate:
    jan max temp ---> VGG16_tradeOps
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)


V0.1A: first working version, 2D input (columns x rows of records) without word encoded.
V0.1B: 2D input with word encoded. Numeric characters are replaced by "X" to reduce the input words
'''





from __future__ import print_function
from __future__ import absolute_import

import warnings
import os
import csv
import numpy as np

import h5py
import pandas
import time
import math

from keras.preprocessing.text import Tokenizer,one_hot,base_filter
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

#from numpy import isnan
from PIL import Image

BASE_DIR = os.getcwd()
#Input files
DATA_PATH = BASE_DIR + '/Data.txt'
LABEL_PATH = BASE_DIR + '/Label.txt'

HEIGHT = 3300
WIDTH = 2550
MAX_NB_WORDS = 20000
OUTPUT = 1
MAX_SEQUENCE_LENGTH = 2
SAMPLE_IMG = BASE_DIR +'/2003ABE-1.png'
Nb_LABELS = 12 + 2 #nan and not defined
NB_DATA_FLD = 5


#Output files
LABEL_BIN_PATH = BASE_DIR + '/Label_B.txt'
WEGITH_FILE = BASE_DIR + '/weight.h5'



###############################################################
#model adopted from iris prediction
def baseline_tradeOps():
#4 inputs -> [4 hidden nodes] -> 1 outputs
	# create model
	model = Sequential()
	model.add(Dense(NB_DATA_FLD, input_dim=NB_DATA_FLD, init='normal', activation='relu'))
	model.add(Dense(Nb_LABELS, init='normal', activation='sigmoid'))
	print('Model loaded.')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('Model compiled.')
	return model
###############################################################
#all numbers are changed...as the exact value does not matter...
def replace_words(word):
        numRep = "X"
        word = word.replace("0",numRep)
        word = word.replace("1",numRep)
        word = word.replace("2",numRep)
        word = word.replace("3",numRep)
        word = word.replace("4",numRep)
        word = word.replace("5",numRep)
        word = word.replace("6",numRep)
        word = word.replace("7",numRep)
        word = word.replace("8",numRep)
        word = word.replace("9",numRep)
        return word
        
###############################################################
#input: various word labels
#output: data and label array set for learning
def data_prep_v1(data_path,label_path,Nb_LABELS):
# no of avaliable labels = 13 (12 months + NA)

        # load dataset
        dataframe = pandas.read_csv(data_path, header=None,delimiter='\t')
        dataset = dataframe.values
        X = dataset[:,1:NB_DATA_FLD].astype('float')
        Y = dataset[:,7].astype('str')
        X_word = dataset[:,0].astype('str')
        encoder = LabelEncoder()    
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)

        rec = len(X_word) # of records in the array

        #convert all the numbers into "X"
        for i in range(rec):
                X_word[i] = replace_words(X_word[i])

        # convert words into sequence
        encoder_word = LabelEncoder()
        encoder_word.fit(X_word)
        encoded_word = encoder_word.transform(X_word)

        print("word_encoded")
        print("encoded_word "+ str(np.shape(encoded_word)))
        print("X "+ str(np.shape(X)))
        #rec = len(encoded_word)
        data_bin = np.zeros((rec,NB_DATA_FLD))
        i=0
        for i in range(rec):
                x_temp = X[i][0:NB_DATA_FLD-1] #one for word, one for index starting 0
                word_temp = encoded_word[i]
                
                if word_temp == "":
                        word_temp = "-"
                #where_are_NaNs = math.isnan(x_temp[0])
                #x_temp[where_are_NaNs] = 0.
                data_bin[i][0] = word_temp
                data_bin[i][1:NB_DATA_FLD] = x_temp
        print("data_bin " + str(np.shape(data_bin)))
        print(data_bin[0])
        return data_bin, dummy_y,encoder,encoder_word

###############################################################
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#retrieve the image
im = Image.open(SAMPLE_IMG)
WIDTH = im.size[0]
HEIGHT = im.size[1]

#prepare the data
#target_label_list = ['maximum temp_1'] #max temperature
data_list,lab_bin_list,encoder,encoder_word = data_prep_v1(DATA_PATH,LABEL_BIN_PATH,Nb_LABELS)
print("data_list:" +str(np.shape(data_list)))
print("lab_bin_list:" +str(np.shape(lab_bin_list)))
print('Data Preprocessed.')

input_length = str(len(data_list))
print("input_length:"+input_length)

#data_list_T = np.transpose(data_list)

#build the model
#output for debug
with open('data_list.txt', 'w') as file:
    file.writelines(str(i) +'\n' for i in data_list)

estimator = KerasClassifier(build_fn=baseline_tradeOps, nb_epoch=200, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test = train_test_split(data_list, lab_bin_list, test_size=0.33, random_state=seed)
print(X_train)
print(Y_train)
print("Fitting Model")
tic = time.clock()
estimator.fit(X_train,Y_train)
toc = time.clock()
print('Model Fit. and took %s' % str(toc))
predictions = estimator.predict(X_test)
print(predictions)
print(encoder.inverse_transform(predictions))
print('Prediction Made.')
estimator.model.save(WEGITH_FILE)
print('Model Saved.')
scores = estimator.model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

