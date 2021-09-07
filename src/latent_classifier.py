#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf
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
###########################################
# latent variable이 분류가능한지 알아보는 코드
# 분류가 잘 된다면, latent variable은 Driving style이 학습되었다고 볼수있다.
# left 데이터셋 (왼쪽차선 주행 데이터셋), right 데이터셋(오른쪽차선 주행 데이터셋)이 있음.
# n100 네트워크는 left 데이터셋만을 사용하여 학습시킴
# n101 네트워크는 right 데이터셋만을 사용하여 학습시킴
# 학습 : 각 네트워크의 latent variable과 네트워크 이름에 대해 label을 주고 Classifier를 학습시킴.
# 테스트 : 학습에 사용한 데이터셋 말고 차선 중앙을 따라가며 수집된 데이터셋을 각 네트워크에 입력하여 latent를 뽑고
# 해당 latent에 대해서도 분류를 잘하는지 확인.

class LatentClassifier:
    def __init__(self, data_path):
        self._generate_data_path(data_path)
        
        self.Config = config.Config
        self.net_model = NetModel(data_path)
        self.drive_data = DriveData(self.csv_path)
        self.image_process = ImageProcess()
        
        self.num_latent = 100
        self.label = []
        self.latent = []
        self.steering_angle = []
        self.label_valid = []
        self.latent_valid = []
        self.steering_angle_valid = []
        self._prepare_data()
        
    def _generate_data_path(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            csv_name = data_path[loc_slash + 1:] # get folder name
        else:
            csv_name = data_path
        self.csv_name = csv_name
        self.csv_path = data_path +'/train'+'/'+csv_name+ '_latent' + const.DATA_EXT  # use it for csv file name
        self.csv_path_valid = data_path +'/valid'+'/'+csv_name+ '_latent' + const.DATA_EXT  # use it for csv file name
        self.data_path = data_path + '/'

    def _drive_data(self):
        self.csv_header = ['label', 'steering_angle']
        for i in range(self.num_latent):
            self.csv_header.insert(i+2,str(i))
            
        self.df = pd.read_csv(self.csv_path, names=self.csv_header, index_col=False)
        self.df_valid = pd.read_csv(self.csv_path_valid, names=self.csv_header, index_col=False)
        num_data = len(self.df)
        num_data_valid = len(self.df_valid)
        # print(num_data)
        
        bar = ProgressBar()
        for i in bar(range(num_data)):
            self.label.append(self.df.loc[i]['label'])
            self.steering_angle.append(self.df.loc[i]['steering_angle'])
            sub_latent=[]
            for n in range(self.num_latent):
                sub_latent.append(self.df.loc[i][str(n)])
            self.latent.append(sub_latent)
        
        bar = ProgressBar()
        for i in bar(range(num_data_valid)):
            self.label_valid.append(self.df_valid.loc[i]['label'])
            self.steering_angle.append(self.df.loc[i]['steering_angle'])
            sub_latent_valid=[]
            for n in range(self.num_latent):
                sub_latent_valid.append(self.df_valid.loc[i][str(n)])
            self.latent_valid.append(sub_latent_valid)
            
        
    def _prepare_data(self):
        self._drive_data()
        num_samples = len(self.label)
        print('Samples: ', num_samples)
        samples = list(zip(self.label, self.latent, self.steering_angle))
        samples_valid = list(zip(self.label_valid, self.latent_valid, self.steering_angle))
        self.train_data = samples
        self.valid_data = samples_valid
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        
    def _prepare_batch_samples(self, batch_samples):
        labels = []
        latents = []
        steering_angles = []
        for label, latent, steering_angle in batch_samples:
            labels.append(label)
            latents.append(latent)
            steering_angles.append(steering_angle)
        
        return labels, latents, steering_angles
    
    def _generator(self, samples, batch_size):
        num_samples = len(samples)
        while True: # Loop forever so the generator never terminates
            samples = sklearn.utils.shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                
                labels, latents, steering_angles = self._prepare_batch_samples(batch_samples)
                # X_train_latent = np.array(latents)
                # X_train_steer = np.array(steering_angles)
                X_train = np.array(latents)
                y_train = np.array(labels)
                
                # print(X_train.shape)
                # print(y_train.shape)
                # print('y : ',y_train)
                yield sklearn.utils.shuffle(X_train, y_train)
    
    # def _mean_squared_error(self, y_true, y_pred):
    #     y_true = tf.Print(y_true, [y_true[0]], "Inside y_true ")
    #     y_pred = tf.Print(y_pred, [y_pred[0]], "Inside y_pred ")
    #     diff = K.abs(y_true - y_pred)
    #     return K.mean(K.square(diff))
    
    def _compile(self):
        learning_rate = self.Config.neural_net['cnn_lr']
        decay = self.Config.neural_net['decay']
        # print(self.Config.neural_net['loss_function'])
        if self.Config.neural_net['loss_function'] == 'mse':
            self.net_model.model.compile(loss=losses.mean_squared_error,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                        metrics=['accuracy'])
        elif self.Config.neural_net['loss_function'] == 'mae':
            self.net_model.model.compile(loss=losses.mean_absolute_error,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                        metrics=['accuracy'])
        elif self.Config.neural_net['loss_function'] == 'sparse_cc':
            self.net_model.model.compile(loss=losses.sparse_categorical_crossentropy,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                        metrics=['accuracy'])
        elif self.Config.neural_net['loss_function'] == 'binary':
            self.net_model.model.compile(loss=losses.binary_crossentropy,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                        metrics=['accuracy'])
    
    def _start_training(self):
        self.train_generator = self._generator(self.train_data, self.Config.neural_net['batch_size'])
        self.valid_generator = self._generator(self.valid_data, self.Config.neural_net['batch_size'])
        
        if (self.train_generator == None or self.valid_generator == None):
            raise NameError('Generators are not ready.')
        
        self.model_name = self.data_path + '_' + self.Config.neural_net_yaml_name \
            + '_N' + str(self.Config.neural_net['network_type'])
        self.model_ckpt_name = self.model_name + '_ckpt'
        ######################################################################
        # checkpoint
        callbacks = []
        
        checkpoint = ModelCheckpoint(self.model_ckpt_name +'.{epoch:02d}-{val_loss:.3f}.h5',
                                     monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)
        
        # early stopping
        patience = self.Config.neural_net['early_stopping_patience']
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, 
                                  verbose=1, mode='min')
        callbacks.append(earlystop)

        self._compile()
        
        self.train_hist = self.net_model.model.fit_generator(
                self.train_generator, 
                steps_per_epoch=self.num_train_samples//self.Config.neural_net['batch_size'], 
                epochs=self.Config.neural_net['num_epochs'], 
                validation_data=self.valid_generator,
                validation_steps=self.num_valid_samples//self.Config.neural_net['batch_size'],
                verbose=1, callbacks=callbacks, 
                use_multiprocessing=True,
                workers=24)
    
    def _train(self):
        self._start_training()
        self.net_model.save(self.model_name)
        
###############################################################################
#
def main():
    if (len(sys.argv) != 2):
        exit('Usage:\n$ python {} data_path'.format(sys.argv[0]))

    gl = LatentClassifier(sys.argv[1])
    gl._train()

###############################################################################
#
if __name__ == '__main__':
    try:
        main()
        
        # t = threading.Thread(target=print, args("multithread Hi",))
        # t.start()
        print("threading start")
        
        # main()
    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')
