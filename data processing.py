import numpy as np
from sklearn import preprocessing

data = np.load('data/vgg_feature.npz')
test_sat_vgg_feature = data['test_sat_vgg_feature']
test_grd_vgg_feature = data['test_grd_vgg_feature']
train_sat_vgg_feature = data['train_sat_vgg_feature']
train_grd_vgg_feature = data['train_grd_vgg_feature']

np.savez('data_vgg/test_sat_vgg_feature.npz', test_sat_vgg_feature=test_sat_vgg_feature)
np.savez('data_vgg/test_grd_vgg_feature.npz',  test_grd_vgg_feature=test_grd_vgg_feature )
np.savez('data_vgg/train_sat_vgg_feature.npz', train_sat_vgg_feature=train_sat_vgg_feature)
np.savez('data_vgg/train_grd_vgg_feature.npz',  train_grd_vgg_feature=train_grd_vgg_feature )

test_sat = preprocessing.normalize(test_sat_vgg_feature)
test_grd = preprocessing.normalize(test_grd_vgg_feature)
train_sat = preprocessing.normalize(train_sat_vgg_feature)
train_grd = preprocessing.normalize(train_grd_vgg_feature)

np.savez('norm_data_vgg/norm_feature.npz', test_sat=test_sat,test_grd=test_grd,
                                           train_sat=train_sat,train_grd=train_grd )

np.savez('norm_data_vgg/test_sat.npz', test_sat=test_sat)
np.savez('norm_data_vgg/test_grd.npz',  test_grd=test_grd )
np.savez('norm_data_vgg/train_sat.npz', train_sat=train_sat)
np.savez('norm_data_vgg/train_grd.npz',  train_grd=train_grd )