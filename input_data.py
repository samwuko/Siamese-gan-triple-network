import cv2
import random
import numpy as np
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


class InputData:
    '''
    This class is mainly used to extract features
    of satellite images and streetview images, and store features in files.
    '''
    img_root = 'splits/'

    def __init__(self):

        self.train_list = self.img_root + 'train-19zl.csv'
        self.test_list = self.img_root + 'val-19zl.csv'

        base_model = VGG16(weights='imagenet')
        self.model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([data[0], data[1], pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)


        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([data[0], data[1], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)




    def test_data_feature_extractor(self):
        test_sat_vgg_feature = np.empty((0, 4096))
        test_grd_vgg_feature = np.empty((0, 4096))
        for i in range(self.test_data_size):
            img_idx = i
            # satellite
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = image.img_to_array(img[:, :, :])
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            fc1 = self.model.predict(img)
            test_sat_vgg_feature = np.concatenate((test_sat_vgg_feature, fc1), axis=0)

            # ground
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][1])
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = image.img_to_array(img[:, :, :])
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            fc1 = self.model.predict(img)
            test_grd_vgg_feature = np.concatenate((test_grd_vgg_feature, fc1), axis=0)
        return test_sat_vgg_feature, test_grd_vgg_feature

    def train_data_feature_extractor(self):
        for i in range(20):
            random.shuffle(self.id_idx_list)
        train_sat_vgg_feature = np.empty((0, 4096))
        train_grd_vgg_feature = np.empty((0, 4096))
        for i in range(self.data_size):
            img_idx = self.id_idx_list[i]
            # satellite
            img = cv2.imread(self.img_root + self.id_list[img_idx][0])
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = image.img_to_array(img[:, :, :])
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            fc1 = self.model.predict(img)
            train_sat_vgg_feature = np.concatenate((train_sat_vgg_feature, fc1), axis=0)
            # ground
            img = cv2.imread(self.img_root + self.id_list[img_idx][1])
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = image.img_to_array(img[:, :, :])
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            fc1 = self.model.predict(img)
            train_grd_vgg_feature = np.concatenate((train_grd_vgg_feature, fc1), axis=0)
        return train_sat_vgg_feature, train_grd_vgg_feature


data = InputData()
test_sat_vgg_feature, test_grd_vgg_feature = data.test_data_feature_extractor()
train_sat_vgg_feature, train_grd_vgg_feature = data.train_data_feature_extractor()

np.savez('data/vgg_feature.npz', test_sat_vgg_feature=test_sat_vgg_feature,
            test_grd_vgg_feature=test_grd_vgg_feature,
         train_sat_vgg_feature=train_sat_vgg_feature,
         train_grd_vgg_feature=train_grd_vgg_feature )


