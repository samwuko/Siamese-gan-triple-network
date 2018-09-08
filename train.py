# -*- coding: utf-8 -*-
"""""
This code uses GANs netwrok for domain adaptation
Loss for discriminator: categorical crossentropy
Loss for generator: matching distance + reconstruciton error 
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.core import Lambda
import keras.backend as K
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# build generator
def build_generator(latent_disc, latent_size):
    model = Sequential()
    model.add(Dense(latent_disc, input_dim=latent_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(latent_disc))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    return model


# build decoder
def Build_Decoder(latent_disc, latent_size):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_disc))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(128, input_dim=latent_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(latent_size))
    model.add(Activation('sigmoid'))
    return model


def discriminator_share(latent_disc):
    '''
    This is part of the shared discriminator and triple model
    '''
    input_data = Input(shape=(latent_disc,))
    x=Dense(128,input_dim=latent_disc, name='dense1')(input_data)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.5)(x)
    x=Dense(128,name='dense2')(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.5)(x)
    model = Model(inputs=[input_data], outputs=[x])
    return model

def l2_norm(vects):
    '''
    l2_normalize the features of A triple_model
    '''
    return K.l2_normalize(vects, axis=1)#(,256)


def l2_norm_output_shape(shapes):
    shape1 = shapes
    return shape1


def build_discriminator_triple(latent_disc, code_length):
    input_data = Input(shape=(latent_disc,))
    x = discriminator_share(latent_disc)(input_data)

    Real_Fake = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=[input_data], outputs=[Real_Fake])

    x_1 = Dense(code_length, activation='relu')(x)
    prob_output = Lambda(l2_norm, output_shape=l2_norm_output_shape)(x_1)
    triple_model = Model(inputs=[input_data], outputs=[prob_output])
    return discriminator, triple_model


# Build triple loss
def siams_distance(vects):
    sat_global, grd_global = vects
    dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
    pos_dist = tf.diag_part(dist_array)  # 
    pair_n = triple_batch_size * (triple_batch_size - 1.0)
    # ground to satellite
    triplet_dist_g2s = pos_dist - dist_array
    loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n
    # satellite to ground
    triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
    loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n
    loss = (loss_g2s + loss_s2g) / 2.0
    return loss

def siams_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], shape1[0])

def siams_loss(y_true, y_pred):
    return K.sum(y_pred)


def validate(grd_descriptor, sat_descriptor):
    '''
    For final result verification
    '''

    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    return accuracy


def save_all_model(generator, AE_Model, discriminator, Similarity_Model, single_triple_model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    generator.save(path+'generator.h5')
    AE_Model.save(path+'AE_Model.h5')
    discriminator.save(path+'discriminator.h5')
    Similarity_Model.save(path+'Similarity_Model.h5')
    single_triple_model.save(path+'single_triple_Model.h5')


# main code
if __name__ == '__main__':

    # Define some important paramters
    epochs = 100
    batch_size = 100
    latent_disc = 128  # input to the discriminator
    latent_size = 4096  # feature extract by VGG
    code_length = 256
    loss_weight = 10.0
    triple_epochs = 150
    triple_batch_size = 100
    save_path = 'model/'
    #Loading training and testing data
    data = np.load('data_vgg/train_sat_vgg_feature.npz')
    train_sat = data['train_sat_vgg_feature']
    data = np.load('data_vgg/train_grd_vgg_feature.npz')
    train_grd = data['train_grd_vgg_feature']
    data = np.load('data_vgg/test_sat_vgg_feature.npz')
    test_sat = data['test_sat_vgg_feature']
    data = np.load('data_vgg/test_grd_vgg_feature.npz')
    test_grd = data['test_grd_vgg_feature']

    # build generator
    generator =build_generator(latent_disc, latent_size)
    latent1 = Input(shape=(latent_size,))
    latent2 = Input(shape=(latent_size,))
    features1 = generator(latent1)
    features2 = generator(latent2)

    # Build matching loss
    def MMD_distance(vects):
        x, y =vects
        h1 = K.mean(x, axis=0)
        h2 = K.mean(y, axis=0)
        return K.mean(K.abs(h1 - h2))

    def MMD_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], shape1[0])
        
    def MMD_loss(y_true, y_pred):
        return K.sum(y_pred)
        
    output_sim = Lambda(MMD_distance,
                        output_shape=MMD_dist_output_shape)([features1, features2])

    Similarity_Model = Model(inputs=[latent1, latent2], outputs=[output_sim])
    #The Similarity_Model seeks to match the distributions of the source and target data in order
    # to confuse the discriminator.
    Similarity_Model.compile(loss=MMD_loss,loss_weights=[1],optimizer=Adam(lr=0.0001))

    # Build Reconstruction loss
    generator.trainable = True
    Decoder = Build_Decoder(latent_disc, latent_size)
    
    Rec_latent2 = Decoder(features2)
    
    AE_Model = Model(inputs=latent2, outputs=Rec_latent2)
    # Define the Generarotr -> Decoder model1 and its input and output
    #That is to constrain the mapping spaces to those
    # that allow a good reconstruction of the original features
    AE_Model.compile(loss='mse', loss_weights=[1], optimizer=Adam(lr=0.0001))

    # Specify the loss and optimization for the discriminator
    discriminator, single_triple_Model = build_discriminator_triple(latent_disc, code_length=code_length)  # Create another instance of the discriminator
    discriminator.compile(loss=['binary_crossentropy'], loss_weights=[1],
                          optimizer=Adam(lr=0.0001))

#############################################################################

    # GAN leanring starts here
    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch + 1, epochs))
        half_batch = int(batch_size / 2)
        nb_batches = int(train_grd.shape[0] / half_batch)

        epoch_gen_loss = []
        epoch_disc_loss = []

        index = 0

        while index < nb_batches:

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of features from source
            grd_data_train=train_grd[index * half_batch:(index + 1) * half_batch] # [0:50]  [50:100]
            grd_data_feat_train = generator.predict(grd_data_train)  # Get the Source output of feature from generator

            idx = np.random.randint(0, train_sat.shape[0], half_batch)
            sat_data_train_random = train_sat[idx]
            sat_data_train_random_feat = generator.predict(sat_data_train_random)

            sat_data_train = train_sat[index * half_batch:(index + 1) * half_batch]
            sat_data_feat_train = generator.predict(sat_data_train)

            # Labels of source and target data second loss [1 0]:   1 real, Fake 0
            # grd :1,sat:0
            labels_RF = np.array([1] * grd_data_train.shape[0] + [0] * half_batch)

            # Concatenate source and target data batches for training Discriminator
            X_batch = np.concatenate((grd_data_feat_train, sat_data_feat_train))

            # Train the discriminator Here:

            disc_loss = discriminator.train_on_batch(X_batch, [labels_RF])
            epoch_disc_loss.append(disc_loss)

            # Used for matching...................
            Not_used_3 = np.array([0] * half_batch)
            # Train the generator
            Loss1=Similarity_Model.train_on_batch([grd_data_train, sat_data_train], Not_used_3)
            # Loss1=0
            # Train decoder
            Loss2=AE_Model.train_on_batch(grd_data_train, grd_data_train)
            Loss3=AE_Model.train_on_batch(sat_data_train, sat_data_train)

            # Compute combined loss
            epoch_gen_loss.append(Loss1+Loss2+Loss3)
            index += 1

        print('\nEpoch {},[Loss_D: {:.8f}, Loss_G: {:.8f}]'.format(epoch + 1, np.mean(epoch_disc_loss), np.mean(epoch_gen_loss)))

    save_all_model(generator, AE_Model, discriminator, Similarity_Model, single_triple_Model, save_path + 'step2/')

################################################################################

    # trian triple network
    sat_tensor = Input(shape=(latent_disc,))
    grd_tensor = Input(shape=(latent_disc,))
    sat_output = single_triple_Model(sat_tensor)
    grd_output = single_triple_Model(grd_tensor)

    output_siams = Lambda(siams_distance, output_shape=siams_dist_output_shape)([sat_output, grd_output])
    triple_Model = Model(inputs=[sat_tensor, grd_tensor], outputs=[output_siams])
    triple_Model.compile(loss=siams_loss, loss_weights=[1], optimizer=Adam(lr=0.0001))

    train_grd_gan_feature = generator.predict(train_grd)
    train_sat_gan_feature = generator.predict(train_sat)
    test_grd_gan_feature = generator.predict(test_grd)
    test_sat_gan_feature = generator.predict(test_sat)

    # triple network leanring starts here
    for triple_epoch in range(triple_epochs):
        print('triple network Epoch {} of {}'.format(triple_epoch + 1, triple_epochs))
        nb_batches = int(train_grd_gan_feature.shape[0] / triple_batch_size)
        print(train_grd_gan_feature.shape[0])
        epoch_triple_loss = []
        index = 0

        while index < nb_batches:

            grd_data_train = train_grd_gan_feature[index * triple_batch_size:(index + 1) * triple_batch_size]
            sat_data_train = train_sat_gan_feature[index * triple_batch_size:(index + 1) * triple_batch_size]
            triple_label = np.array([0] * grd_data_train.shape[0])

            triple_loss = triple_Model.train_on_batch([sat_data_train, grd_data_train], triple_label)
            epoch_triple_loss.append(triple_loss)
            index += 1

        print('\n[Loss_triple: {:.8f}'.format(np.mean(epoch_triple_loss)))
        sat_output1 = single_triple_Model.predict(test_sat_gan_feature)
        grd_output1 = single_triple_Model.predict(test_grd_gan_feature)
        # test accuracy
        print('compute accuracy')
        te_acc = validate(grd_output1, sat_output1)
        with open(save_path+'accuracy.txt', 'a') as file:
            file.write(str(triple_epoch) +' : ' + str(te_acc) + '\r\n')
        print('%d:* Accuracy on test set: %0.8f%%' % (triple_epoch, 100 * te_acc))
        model_dir = save_path + 'tri_Model/' + str(triple_epoch) + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        triple_Model.save(model_dir + 'model.h5')
        single_triple_Model.save(save_path+'single_triple_Model.h5')

    save_all_model(generator, AE_Model, discriminator, Similarity_Model, single_triple_Model, save_path + '/')
    triple_Model.save(save_path + 'triple_Model.h5')
