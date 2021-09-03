#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from keras.models import model_from_json, Sequential
import multiprocessing
## import oscar ###########################
sys.path.insert(0, './oscar/neural_net/*')
from net_model import NetModel
from drive_data import DriveData
from image_process import ImageProcess
import const
import config
###########################################

class GetLatent:
    def __init__(self, data_path, model_path):
        #latent variable를 얻기위한 코드
        # 서로다른 스타일의 LSTM 네트워크 모델에 이미지데이터셋을 넣고 나온 결과 (30,1)와 label을 짝지은 데이터 구하는 코드
        self._generate_data_path(data_path)
        
        self.Config = config.Config
        self.net_model = NetModel(model_path)
        self.drive_data = DriveData(self.csv_path)
        self.image_process = ImageProcess()
        
        self.images = []
        self.latent_var = []
        self.measurement = []
        self.image_name = []

        self._get_latent()
        self._export_data()
        
    def _generate_data_path(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            csv_name = data_path[loc_slash + 1:] # get folder name
        else:
            csv_name = data_path
        self.csv_name = csv_name
        self.csv_path = data_path + '/' + csv_name + const.DATA_EXT  # use it for csv file name
        self.data_path = data_path + '/'
    
    def _prepare_data(self):
        self.drive_data.read()
        self.test_data = list(zip(self.drive_data.image_names, self.drive_data.measurements))
        num_samples = len(self.test_data)
        print('Samples: ', num_samples)

        lstm_timestep = self.Config.neural_net['lstm_timestep']
        lstm_dataterm = self.Config.neural_net['lstm_dataterm']
        
        self.iter_lstm = (num_samples - lstm_timestep*lstm_dataterm)//lstm_dataterm
        
        image_names = []
        measurements = []
        images = []

        sub_image_names = []
        sub_images = []
        for i, (image_name, measurement) in enumerate(self.test_data):
            image_path = self.data_path + image_name
            image = cv2.imread(image_path)
            if self.Config.data_collection['crop'] is not True:
                image = image[self.Config.data_collection['image_crop_y1']:self.Config.data_collection['image_crop_y2'],
                                self.Config.data_collection['image_crop_x1']:self.Config.data_collection['image_crop_x2']]
            image = cv2.resize(image,
                                (self.Config.neural_net['input_image_width'], self.Config.neural_net['input_image_height']))
            image = self.image_process.process(image)
            
            sub_images.append(image)
            sub_image_names.append(image_name)
            if len(sub_images) > lstm_timestep:
                del sub_images[0]
                del sub_image_names[0]
                temp_sub_images = list(sub_images)
                temp_sub_names = list(sub_image_names)
                images.append(temp_sub_images)
                image_names.append(temp_sub_names)
                measurements.append(measurement)
            cur_output = 'Prepare data : {0}/{1}\r'.format(i, num_samples)
            sys.stdout.write(cur_output)
            sys.stdout.flush()
        
        self.test_lstm_data = list(zip(images, image_names, measurements))

    def _get_latent(self):
        self._prepare_data()
        
        model = self.net_model.model
        
        latent_model = Sequential()
        for i in range(len(model.layers)):
            # print(model.layers[i].output)
            latent_model.add(model.layers[i])
            if model.get_layer(index = i).name is 'lstm':
                break
        latent_model.summary()
        for i in range(self.iter_lstm):
            npimg = np.expand_dims(self.test_lstm_data[i][0], axis=0).reshape(-1, 
                                                            self.Config.neural_net['lstm_timestep'], 
                                                            self.Config.neural_net['input_image_height'],
                                                            self.Config.neural_net['input_image_width'],
                                                            self.Config.neural_net['input_image_depth'])
            # print(npimg.shape)
            latent_var = latent_model.predict(npimg)
            # print(latent_var)
            self.latent_var.append(latent_var)
            
            cur_output = 'Get latent : {0}/{1}\r'.format(i, self.iter_lstm)
            sys.stdout.write(cur_output)
            sys.stdout.flush()
        
        # self.latent_var.append()
        
    def _export_data(self):
        
        text = open(str(self.data_path) 
                    + str(self.csv_name)
                    + '_latent'
                    + '_n' +  str(self.Config.neural_net['network_type']) 
                    + const.DATA_EXT, "w+")
        
        for i in range(self.iter_lstm):
            # print(self.measurement[i][0])
            latent = ""
            image_name = ""
            for j in range(len(self.latent_var[i][0])):
                latent += str(self.latent_var[i][0][j])
                if j is not len(self.latent_var[i][0])-1:
                     latent += ', '
            # print(self.test_lstm_data[i][1])
            for j in range(len(self.test_lstm_data[i][1])):
                image_name += str(self.test_lstm_data[i][1][j])
                # print(len(self.image_name[i][j]))
                if j is not len(self.test_lstm_data[i][1])-1:
                     image_name += ', '
            # print(self.measurement)
            line = "{},{},{}\r\n".format( image_name, 
                                            self.test_lstm_data[i][2][0], 
                                            latent)
            text.write(line)

            cur_output = 'Export latent : {0}/{1}\r'.format(i, self.iter_lstm)
            sys.stdout.write(cur_output)
            sys.stdout.flush()

###############################################################################
#
def main():
    if (len(sys.argv) != 3):
        exit('Usage:\n$ python {} data_path load_model_path'.format(sys.argv[0]))

    gl = GetLatent(sys.argv[1], sys.argv[2])


###############################################################################
#
if __name__ == '__main__':
    try:
        pool = multiprocessing.Pool(processes=10)
        pool.map(main())
        pool.close()
        pool.join()
        
        # t = threading.Thread(target=print, args("multithread Hi",))
        # t.start()
        print("threading start")
        
        # main()
    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')
