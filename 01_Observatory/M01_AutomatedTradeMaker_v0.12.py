# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

This program attempts to implement the previous examples in the context of document recognition
Option 1. Word lists with numeric variables are put it for testing if it is the target label

Option 2. Images are fed to the VGG net for deciding the target coordinate:
    jan max temp ---> VGG16_tradeOps
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

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
Nb_LABELS = 13
NB_DATA_FLD = 4


#Output files
LABEL_BIN_PATH = BASE_DIR + '/Label_B.txt'
WEGITH_FILE = BASE_DIR + '/weight.h5'




###############################################################
def VGG16_tradeOpsv01(Image_Nb, HEIGHT, WIDTH, OUTPUT_DUMMY, NB_DATA_FLD, include_top=True, input_tensor=None):
#in_length, not used...to be removed later...

    # Determine proper input shape
    #th to be amended...
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (NB_DATA_FLD, WIDTH, HEIGHT)
        else:
            input_shape = (NB_DATA_FLD, None, None)
    else: #tf
        #Input shape
        #4D tensor with shape: (samples, channels, rows, cols) if dim_ordering='th' or 4D tensor with shape: (samples, rows, cols, channels) if dim_ordering='tf'.
        #1D for now...
        if include_top: #include top
            #input_shape = (None, HEIGHT, WIDTH,  NB_DATA_FLD)
            input_shape = (1,NB_DATA_FLD)
            print('parameter: top,tf')
        else:
            input_shape = (None, None, None, NB_DATA_FLD)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
            print('parameter: keras tensor ' + str(img_input))
        else:
            img_input = input_tensor
            print('parameter: non-keras tensor')
       
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1',input_shape=(1,NB_DATA_FLD))(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(OUTPUT_DUMMY, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(input=img_input, output=x)

    return model
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
#input: various word labels
#output: data and label array set for learning
def data_prep_v1(data_path,label_path,Nb_LABELS):
# no of avaliable labels = 13 (12 months + NA)


    # load dataset
    dataframe = pandas.read_csv(data_path, header=None,delimiter='\t')
    dataset = dataframe.values
    X = dataset[:,1:5]
    Y = dataset[:,7]
    label_list=[]
    word_list = []

    data_lines = open(data_path,'r').readlines()
    for line in data_lines:
        words = line.split("\t")
        if len(words)== 8 : #must have values...can't be
            #accumulate text for tokenization later
            word_list.append(words[0])
            #data_list.append(words[1:5])
            #data_arr.concatenate(data_arr,words[1:5])
            label_txt = words[7].strip()
            #label_list.append(label_txt)
            #Y.append(label_txt)

    #dummy_y = one_hot(Y, Nb_LABELS,filters=base_filter())
    #dummy_y = one_hot(label_list, Nb_LABELS,filters=base_filter())
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    #output the binned label
    print(dummy_y)
    #output for debug
    #with open(label_path, 'w') as file:
    #    file.writelines(str(i) +'\n' for i in dummy_y)

    #tokenize words recognized:
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(word_list)
    sequences = tokenizer.texts_to_sequences(word_list)
    tokenized_text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    data_bin = []
    ##to be checked if data_bin can be outputed...
    #word tokenized and add the other features of the words
    #for word, X_word  in zip(tokenized_text,X):
    #    tobeAppended = word+X_word
    #    data_bin.append(tobeAppended)
        
    #return data_bin,dummy_y
    return X, dummy_y,encoder
###############################################################
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#retrieve the image
im = Image.open(SAMPLE_IMG)
WIDTH = im.size[0]
HEIGHT = im.size[1]


#prepare the data
#file_Nb ==> no of different images
#data, label,Image_Nb= data_prep(WIDTH,HEIGHT)

#target_label_list = ['maximum temp_1'] #max temperature
data_list,lab_bin_list,encoder = data_prep_v1(DATA_PATH,LABEL_BIN_PATH,Nb_LABELS)
print('Data Preprocessed.')

input_length = len(data_list)

#build the model
#stake up the tensorz
#BATCH_SIZE = 10
#input_t = K.zeros(shape=(input_length, MAX_SEQUENCE_LENGTH))
#print('Input shape: ' + str(input_t))
#Image_Nb = 0 #not in use for 1D structure
#model = VGG16_tradeOps(Image_Nb, HEIGHT, WIDTH, OUTPUT_DUMMY, NB_DATA_FLD, input_tensor = input_t)

#output for debug
with open('data_list.txt', 'w') as file:
    file.writelines(str(i) +'\n' for i in data_list)

estimator = KerasClassifier(build_fn=baseline_tradeOps, nb_epoch=200, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test = train_test_split(data_list, lab_bin_list, test_size=0.33, random_state=seed)
#print(X_train)
#print(Y_train)
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
scores = estimator.model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

