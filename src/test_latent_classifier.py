#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import pandas as pd
from progressbar import ProgressBar
from keras import losses, optimizers
from keras.models import model_from_json, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import sklearn
from sklearn.model_selection import train_test_split
import multiprocessing
## import oscar ###########################
sys.path.insert(0, './oscar/neural_net/*')
from net_model import NetModel
from drive_data import DriveData
from image_process import ImageProcess
import const
import config

class TestLatentClassifier:
    def __init__(self, model_path, data_path):
        self._generate_data_path(data_path)
        
        self.Config = config.Config
        self.drive_data = DriveData(self.csv_path)
        self.image_process = ImageProcess()
        
        self.num_latent = 100
        self.label = []
        self.latent = []
        self._load(model_path)
        
    def _generate_data_path(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        if data_path[-3:] == 'csv':
            loc_slash = data_path.rfind('/')
            if loc_slash != -1: # there is '/' in the data path
                csv_name = data_path[loc_slash + 1:] # get folder name
            self.csv_name = csv_name[:-4]
            self.csv_path = data_path
            # self.data_path = data_path
        else:
            loc_slash = data_path.rfind('/')
            if loc_slash != -1: # there is '/' in the data path
                csv_name = data_path[loc_slash + 1:] # get folder name
            else:
                csv_name = data_path
            self.csv_name = csv_name
            self.csv_path = data_path + '/' + csv_name + '_latent' + const.DATA_EXT  # use it for csv file name
            # self.data_path = data_path + '/'

    def _drive_data(self):
        self.csv_header = ['label', 'steering_angle']
        for i in range(self.num_latent):
            self.csv_header.insert(i+2,str(i)) # add 'latent_number'
            
        self.df = pd.read_csv(self.csv_path, names=self.csv_header, index_col=False)
        num_data = len(self.df)
        bar = ProgressBar()
        for i in bar(range(num_data)):
            self.label.append(self.df.loc[i]['label'])
            
            sub_latent=[]
            for n in range(self.num_latent):
                sub_latent.append(self.df.loc[i][str(n)])
            self.latent.append(sub_latent)
            
    def _prepare_batch_samples(self, batch_samples):
        labels = []
        latents = []
        for label, latent in batch_samples:
            labels.append(label)
            latents.append(latent)
        
        return labels, latents
    
    def _compile(self):
        learning_rate = self.Config.neural_net['cnn_lr']
        decay = self.Config.neural_net['decay']
        # print(self.Config.neural_net['loss_function'])
        if self.Config.neural_net['loss_function'] == 'mse':
            self.model.compile(loss=losses.mean_squared_error,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                        metrics=['accuracy'])
        elif self.Config.neural_net['loss_function'] == 'mae':
            self.model.compile(loss=losses.mean_absolute_error,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                        metrics=['accuracy'])
        elif self.Config.neural_net['loss_function'] == 'sparse_cc':
            self.model.compile(loss=losses.sparse_categorical_crossentropy,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                        metrics=['accuracy'])
        elif self.Config.neural_net['loss_function'] == 'binary':
            self.model.compile(loss=losses.binary_crossentropy,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                        metrics=['accuracy'])
    
    def _load(self, model_path):
        self.model = model_from_json(open(model_path+'.json').read())
        self.model.load_weights(model_path+'.h5')
        self._compile()
    
    def _test(self):
        self._drive_data()
        self.test_data = list(zip(self.label, self.latent))
        num_samples = len(self.test_data)
        print('Samples: ', num_samples)
        correct_count = 0
            
        labels, latents = self._prepare_batch_samples(self.test_data)
        latents = np.array(latents)
        gt_label = np.array(labels)
        
        # model = self.net_model.model
        
        # latent_model = Sequential()
        # for i in range(len(model.layers)):
        #     # print(model.layers[i].output)
        #     latent_model.add(model.layers[i])
        #     if model.get_layer(index = i).name in 'dropout':
        #         print('dropout')
        #         # break
        # latent_model.summary()
        for i in range(len(self.test_data)):
            # print('latent: ', latents[i])
            latent = latents[i][:]
            predict = (self.model.predict(np.expand_dims(latent, axis=0))>0.5).astype('int32')[0][0]
            # print(predict)
            print(str(predict) + " " + str(int(gt_label[i])))
            # print(str(np.argmax(predict)) + " " + str(int(gt_label[i])))
            if predict == int(gt_label[i]):
                correct_count += 1
                    
        print("correct_count : "+ str(correct_count)+ "/"+ str(num_samples))
            
        
###############################################################################
#
def main():
    if (len(sys.argv) != 3):
        exit('Usage:\n$ python {} model_path, data_path'.format(sys.argv[0]))

    gl = TestLatentClassifier(sys.argv[1], sys.argv[2])
    gl._test()

###############################################################################
#
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')
