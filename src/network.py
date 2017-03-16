from features import sigproc
import numpy as np
import time
from sklearn import datasets
import pickle
import pdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,\
    Convolution1D
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from process import *
import os
from keras import callbacks
from keras.layers.normalization import BatchNormalization
def encodeLabels(label):
    #print 'THE label is',label
    if label=='angry':
        return np.array([1,0,0,0,0,0])
    if label=='fear':
        return np.array([0,1,0,0,0,0])
    if label=='happy':
        return np.array([0,0,1,0,0,0])
    if label=='neutral':
        return np.array([0,0,0,1,0,0])
    if label=='sad':
        return np.array([0,0,0,0,1,0])
    if label=='surprise':
        return np.array([0,0,0,0,0,1])
    return np.array([0,0,0,1,0,0])

def decodeLabels(arrayLabel):
    index = np.where(arrayLabel==1)
    return int(index[0])
#import pdb

def maintrainnet():
    tupledata=processwav()    
    for i  in np.arange( len(tupledata)  ):      
        tupledata[i][0]=tupledata[i][0][0:40]
        
    sessionTrain, sessionTest = train_test_split(tupledata, test_size=0.1)
    start = time.time()
    window=2
    allUt = []
    allLabels = []
    utteranceByFeat = []
    labelsByFeat = []
    portionSelection=1.0
    for utterance in (sessionTrain):
        allUt.append([utterance[0]])        
        allLabels.append(encodeLabels(utterance[1]))
    print '----SHAPE--- ',np.array(allUt).shape

    testUtterance = []
    X_test = []
    Y_test = []
    #X_train=np.array()
    for utterance in (sessionTest):
        X_test.append([utterance[0]])
        Y_test.append(encodeLabels(utterance[1]))    
   
    end = time.time()   
    print 'segment level feature extraction total time: ' +str(end - start)
    print 'building middle layer'
    
    #nb_filters = 100
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 2
    voice_rows=240
    voice_cols=100
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3    
    model=Sequential()
   
    model.add(
              Convolution2D( 20, 3, 3, border_mode='same',input_shape=(1, 40, 40) )
              )
    model.add(Activation('relu'))
    
    #model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Convolution2D(10,5,5))
    model.add(Dropout(0.25))
    
    
    model.add(Convolution2D( 40, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #model.add(Convolution2D(20, 5, 5))
    
    model.add(Convolution2D(60, 3, 3, border_mode='same'))
    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))    
    
    model.add(Convolution2D(80, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(80*2*2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation('softmax')) 
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
    
    #print 'allUt shape is ',np.array(allUt).shape
    X_train=np.array(allUt)
    X_test=np.array(X_test)
    print 'shape is ',X_test.shape
    X_train = X_train.reshape(X_train.shape[0], 1, 40, 40)
    X_train = X_train.astype('float32')
    #X_test = X_test.reshape(X_test.shape[0], 1, 40, 40)
    X_test = X_test.astype('float32')
   
    print 'X_train.shape IS ',X_train.shape
    print 'X_test.shape IS ',X_test.shape

    model.fit(X_train, np.array(allLabels),
          validation_data=(X_test, np.array(Y_test)), nb_epoch=50, batch_size=40,show_accuracy=True, shuffle=True)

    print 'estimating emotion state prob. distribution'
    stateEmotions = []
    #score = model.evaluate(X_test, np.array(Y_test), show_accuracy=False, verbose=0)
    score = model.evaluate(X_test, np.array(Y_test), show_accuracy=True, verbose=0)
    print 'Test scores is ',score

    
maintrainnet()    
    