#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
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

class TransferLearningLSTM:
    def __init__(self, model_path, data_path):
        self._generate_data_path(data_path)
        self.Config = config.Config
        # self.net_model = NetModel(data_path)
        # self.drive_data = DriveData(self.csv_path)
        self.t_data = DriveData(data_path+'/train/'+ self.model_name+'/'+ self.model_name + const.DATA_EXT)
        self.v_data = DriveData(data_path+'/valid/'+ self.model_name+'/'+ self.model_name + const.DATA_EXT)
        self.t_data_path = data_path+'/train/'+ self.model_name
        self.v_data_path = data_path+'/valid/'+ self.model_name
        self.image_process = ImageProcess()
        self.model_path = model_path
        self.train_data = []
        self.valid_data = []
        self.model_name = data_path + '_' + self.Config.neural_net_yaml_name \
            + '_N' + str(self.Config.neural_net['network_type'])
        self._prepare_data()
        
    def _generate_data_path(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            self.model_name = data_path[loc_slash + 1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            self.model_name = data_path
        self.csv_path = data_path + '/' + self.model_name + const.DATA_EXT  # use it for csv file name 


    def _prepare_data(self):
        self.t_data.read()
        self.v_data.read()
        # put velocities regardless we use them or not for simplicity.
        train_samples = list(zip(self.t_data.image_names, self.t_data.velocities, self.t_data.measurements))
        valid_samples = list(zip(self.v_data.image_names, self.v_data.velocities, self.v_data.measurements))
        self.train_data = self._prepare_lstm_data(train_samples)
        self.valid_data = self._prepare_lstm_data(valid_samples)
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        print('Train samples: ', self.num_train_samples)
        print('Valid samples: ', self.num_valid_samples)

    def _prepare_lstm_data(self, samples):
        num_samples = len(samples)
        # get the last index number
        last_index = (num_samples - self.Config.neural_net['lstm_timestep']*self.Config.neural_net['lstm_dataterm'])
        image_names = []
        measurements = []
        velocities = []
        for i in range(0, last_index):
            timestep_samples = samples[ i : i+self.Config.neural_net['lstm_timestep']*self.Config.neural_net['lstm_dataterm'] :self.Config.neural_net['lstm_dataterm']]
            timestep_image_names = []
            timestep_measurements = []
            timestep_velocities = []
            for image_name, velocity, measurment in timestep_samples:
                timestep_image_names.append(image_name)
                timestep_measurements.append(measurment)
                timestep_velocities.append(velocity)
            image_names.append(timestep_image_names)
            measurements.append(timestep_measurements)
            velocities.append(timestep_velocities)
        data = list(zip(image_names, velocities, measurements))
        return sklearn.utils.shuffle(data)
    
    def _prepare_lstm_batch_samples(self, batch_samples, data=None):
        images = []
        velocities = []
        measurements = []
        if data == 'train':
            data_path = self.t_data_path
        elif data == 'valid':
            data_path = self.v_data_path
        for i in range(0, self.Config.neural_net['batch_size']):
            images_timestep = []
            image_names_timestep = []
            velocities_timestep = []
            measurements_timestep = []
            for j in range(0, self.Config.neural_net['lstm_timestep']):
                image_name = batch_samples[i][0][j]
                image_path = data_path + '/' + image_name
                image = cv2.imread(image_path)
                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if self.Config.data_collection['crop'] is not True:
                    image = image[self.Config.data_collection['image_crop_y1']:self.Config.data_collection['image_crop_y2'],
                                self.Config.data_collection['image_crop_x1']:self.Config.data_collection['image_crop_x2']]
                image = cv2.resize(image, 
                                (self.Config.neural_net['input_image_width'],
                                self.Config.neural_net['input_image_height']))
                image = self.image_process.process(image)
                images_timestep.append(image)
                image_names_timestep.append(image_name)
                velocity = batch_samples[i][1][j]
                velocities_timestep.append(velocity)
                
                if j is self.Config.neural_net['lstm_timestep']-1:
                    measurement = batch_samples[i][2][j]
                    # if no brake data in collected data, brake values are dummy
                    steering_angle, throttle, brake = measurement
                    
                    if abs(steering_angle) < self.Config.neural_net['steering_angle_jitter_tolerance']:
                        steering_angle = 0
                        
                    if self.Config.neural_net['num_outputs'] == 2:                
                        measurements_timestep.append((steering_angle*self.Config.neural_net['steering_angle_scale'], throttle))
                    else:
                        measurements_timestep.append(steering_angle*self.Config.neural_net['steering_angle_scale'])
            
            images.append(images_timestep)
            velocities.append(velocities_timestep)
            measurements.append(measurements_timestep)

        return images, velocities, measurements
    
    def _generator(self, samples, batch_size, data):
        num_samples = len(samples)
        while True: # Loop forever so the generator never terminates
            samples = sklearn.utils.shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                
                images, velocities, measurements = self._prepare_lstm_batch_samples(batch_samples, data)
                # X_train_latent = np.array(latents)
                # X_train_steer = np.array(steering_angles)
                X_train = np.array(images)
                y_train = np.array(measurements)
                
                # print(X_train.shape)
                # print(y_train.shape)
                # print('y : ',y_train)
                yield sklearn.utils.shuffle(X_train, y_train)
        
    def _load(self):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(self.Config.neural_net['gpus'])
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.tensorflow_backend.set_session(sess)
        self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        # self.model.summary()
        for i in range(len(self.model.layers)):
            if self.model.get_layer(index = i).name == 'lstm':
                print(self.model.get_layer(index = i).name)
                self.model.layers[i].trainable = True
            else:
                self.model.layers[i].trainable = False
        # for i in range(len(self.model.layers)):
        #     print(self.model.layers[i].trainable)
        self.model.summary()
        self._compile()
    
    def _compile(self):
        learning_rate = self.Config.neural_net['cnn_lr']
        decay = self.Config.neural_net['decay']
        
        self.model.compile(loss=losses.mean_squared_error,
                    optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                    metrics=['accuracy'])
    
    def _start_training(self):
        self._load()
        self.train_generator = self._generator(self.train_data, self.Config.neural_net['batch_size'], data='train')
        self.valid_generator = self._generator(self.valid_data, self.Config.neural_net['batch_size'], data='valid')
        
        if (self.train_generator == None or self.valid_generator == None):
            raise NameError('Generators are not ready.')
        
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

        
        self.train_hist = self.model.fit_generator(
                self.train_generator, 
                steps_per_epoch=self.num_train_samples//self.Config.neural_net['batch_size'], 
                epochs=self.Config.neural_net['num_epochs'], 
                validation_data=self.valid_generator,
                validation_steps=self.num_valid_samples//self.Config.neural_net['batch_size'],
                verbose=1, callbacks=callbacks, 
                use_multiprocessing=True,
                workers=1)
    
    def _train(self):
        self._start_training()
        self.model.save(self.model_name)
        
###############################################################################
#
def main():
    if (len(sys.argv) != 3):
        exit('Usage:\n$ python {} model_path data_path'.format(sys.argv[0]))

    gl = TransferLearningLSTM(sys.argv[1], sys.argv[2])
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
