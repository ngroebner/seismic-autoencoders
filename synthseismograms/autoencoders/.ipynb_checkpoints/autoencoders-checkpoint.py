'''
Contains multiple types of autoencoders used for clustering seismograms and spectrograms.

Includes 
A. Autoencoder structures
    1. Causal (temporal) convolutional networks - symmetric
    2. Causal convolutional networks with skip connections (similar to Wavenet) - NOT IMPLEMENTED
    3. Simple 2D convolutional networks (for spectrograms only) - symmetric

B. Clustering methods - can use the above structures
    1. Deep Convolutional Embedded Clustering
    2. Simple K-means on latent vectors
    3. Variational Deep Embedding - NOT IMPLEMENTED
'''


from keras.layers import Input, Dense, Dropout, Flatten, Reshape, Add, Activation
from keras.layers import Conv2D, Conv1D

from .layers import *
from keras.models import Sequential, Model
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils.vis_utils import plot_model

from keras import optimizers
from keras.utils import multi_gpu_model

from time import time


import autoencoders.metrics as metrics

import numpy as np

from sklearn.cluster import KMeans

ALLOWED_NNET_TYPES = ['1D', '2D']

#NEED TO IMPLEMENT SKIP CONNECTIONS FOR ALL OF THESE

def CausalCAE(nnet_type, latent_dims, input_shape, n_filters=32, dilation_rate=2, n_layers=4, use_skip_connections=False):

    assert nnet_type in ALLOWED_NNET_TYPES, print('nnet_type must be one of {}'.format(ALLOWED_NNET_TYPES))

    if nnet_type == '1D': Conv = Conv1D
    if nnet_type == '2D': Conv = Conv2D

    # ENCODER
    input_ = Input(shape=input_shape, name='input')
    x = input_

    skips = []
    for i in [dilation_rate**i for i in range(n_layers-1)]:
        x = Conv(n_filters, 2, dilation_rate=i, activation='relu', padding='causal', name='conv{}'.format(i))(x)
        skips.append(x)

    # last layer constructed as follows to accomodate skip connections
    final_dilation_rate = dilation_rate**(n_layers-1)
    x = Conv(n_filters, 2, dilation_rate=final_dilation_rate, padding='causal', name='conv{}'.format(final_dilation_rate))(x)
    skips.append(x)

    if use_skip_connections:
        x = Add(name='add_encoder_skip_connections')(skips)
    x = Activation('relu')(x)

    encode = Flatten(name='flatten')(x)
    encode = Dense(latent_dims, name='embedding')(encode)
    encode = Reshape((1,latent_dims), name='reshaped_embedding')(encode)

    # DECODER
    decode = Dense(input_shape[0], activation='relu',name='flatten2')(encode)
    decode = Reshape((input_shape[0],1),name='reshape2')(decode)


    skips = []
    for i in [dilation_rate**i for i in range(n_layers)][::-1][:-1]:
        #last step should be a dilatation of 1, skip this to put a sigmoid activation on last layer
        decode = Conv(n_filters, 2, dilation_rate=i,activation='relu', padding='causal',name='deconv{}'.format(i))(decode)
        skips.append(decode)

    # last layer constructed as follows to accomodate skip connections
    decode = Conv(n_filters, 2, dilation_rate=1, padding='causal',name='deconv1')(decode)
    skips.append(decode)

    if use_skip_connections:
        decode = Add(name='add__decoder_skip_connections')(skips)
    decode = Activation('sigmoid')(decode)
    decode = Dense(input_shape[1],name='dense_last')(decode)

    model = Model(input_, decode)

    return model


def CAE(nnet_type, latent_dims, input_shape, n_filters=32, n_layers=4, use_skip_connections=False):

    assert nnet_type in ALLOWED_NNET_TYPES, print('""nnet_type must be one of '.format(ALLOWED_NNET_TYPES))

    if nnet_type == '1D': Conv = Conv1D
    if nnet_type == '2D': Conv = Conv2D

    # ENCODER
    input_ = Input(shape=input_shape, name='input')
    x = input_

    skips = []
    for i in range(n_layers-1):
        print(i)
        x = Conv(n_filters, 2, activation='relu', padding='same', name='conv{}'.format(i))(x)
        skips.append(x)

    # last layer constructed as follows to accomodate skip connections
    x = Conv(n_filters, 2, padding='same', name='conv{}'.format((n_layers-1)**2))(x)
    skips.append(x)

    if use_skip_connections:
        x = Add(name='add_encoder_skip_connections')(skips)
    x = Activation('relu')(x)
    encode = Flatten(name='flatten')(x)
    encode = Dense(latent_dims, name='embedding')(encode)
    encode = Reshape((1,latent_dims), name='reshaped_embedding')(encode)

    # DECODER
    decode = Dense(input_shape[0], activation='relu',name='flatten2')(encode)
    decode = Reshape((input_shape[0],1),name='reshape2')(decode)


    skips = []
    for i in range(n_layers-1):
        #last step should be a dilatation of 1, skip this to put a sigmoid activation on last layer
        decode = Conv(n_filters, 2, activation='relu', padding='same', name='deconv{}'.format(i))(decode)
        skips.append(decode)

    # last layer constructed as follows to accomodate skip connections
    decode = Conv(n_filters, 2, padding='same', name='deconv_last')(decode)
    skips.append(decode)

    if use_skip_connections:
        decode = Add(name='add__decoder_skip_connections')(skips)
    decode = Activation('sigmoid')(decode)
    decode = Dense(input_shape[1],name='dense_last')(decode)

    model = Model(input_, decode)

    return model


class DCEC(object):

    '''
    Takes an uncompiled autoencoder model that has a layer named 'embedding'.
    This model lives in self.cae.  A vanilla kmeans clustering can be performed on the results
    of applying self.extract_feature to a dataset, after applying self.pretrain to that dataset.
    '''
    def __init__(self,
                model,
                n_clusters=4,
                use_skip_connections=False,
                input_shape,
                n_filters=32,
                n_layers=4,
                lamda=0.1,
                multi_gpu=False):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.lamda = lamda
        self.pretrained = False
        self.y_pred = []
        self.multi_gpu = multi_gpu
        
        if model =='causal':
            self.cae = CausalCAE('1D', n_clusters, 
                                 input_shape, n_filters=n_filters, 
                                 dilation_rate=2, n_layers=n_layers, 
                                 use_skip_connections=use_skip_connections)
        elif model == 'acausal':
            self.cae = CAE('1D', n_clusters, 
                                 input_shape, n_filters=n_filters, 
                                 n_layers=n_layers, 
                                 use_skip_connections=use_skip_connections)
        else:
            self.cae = model

        hidden = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self._model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output])

        if multi_gpu == True:
            try:
                self.model = multi_gpu_model(self._model, cpu_relocation=True)
                print("**************** Training using multiple GPUs.. ****************")
            except ValueError as e:
                self.model = self._model
                print(e)
                print("**************** Training using single GPU or CPU.. ****************")
        else:
            self.model = self._model
            print("**************** Training using single GPU or CPU.. ****************")

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam', save_dir='results/temp'):
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(save_dir + '/pretrain_log.csv')

        # begin training
        t0 = time()
        history = self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        print('Pretraining time: ', time() - t0)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True
        return history

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=[self.lamda, (1-self.lamda)], optimizer=optimizer)

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3,
            update_interval=140, save_interval=None, cae_weights=None, save_dir='./results/temp'):

        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        history = {'ite':[], 'acc':[], 'nmi':[], 'ari':[], 'L':[], 'Lc':[], 'Lr':[]}

        print('Update interval', update_interval)
        if save_interval is None:
            save_interval = x.shape[0] / batch_size * 5
        print('Save interval', save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            pretrain_history = self.pretrain(x, batch_size, save_dir=save_dir)
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
            print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        # use self._model because of multigpu?
        self._model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        '''logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()'''

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    history['ite'].append(ite)
                    history['acc'].append(acc)
                    history['nmi'].append(nmi)
                    history['ari'].append(ari)
                    history['L'].append(loss[0])
                    history['Lc'].append(loss[1])
                    history['Lr'].append(loss[2])
                    
                    '''logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)'''
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss(L,Lc,Lr)=', loss)
                else:
                    loss = np.round(loss, 5)
                    history['ite'].append(ite)
                    history['L'].append(loss[0])
                    history['Lc'].append(loss[1])
                    history['Lr'].append(loss[2])
                    '''logdict = dict(iter=ite, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)'''
                    print('Iter', ite, '; loss(L,Lc,Lr)=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    #logfile.close()
                    break

            #print losses every iteration, but only update target distribution every update_interval iterations      
            '''loss = np.round(loss, 5)
            logdict = dict(iter=ite, L=loss[0], Lc=loss[1], Lr=loss[2])
            logwriter.writerow(logdict)
            print('Iter', ite, '; loss=', loss)'''

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                #print("Training on batch")
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                #print("Training on batch (index + 1)")
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            '''if ite % save_interval == 0:
                # save DCEC model checkpoints
                print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')'''

            ite += 1

        # save the trained model
        #logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.h5')
        # use self._model because self.model might be multi gpu
        if self.multi_gpu == True: self._model.save(save_dir + '/dcec_model_final.h5')
        else: self.model.save(save_dir + '/dcec_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)
        
        return [history, pretrain_history]


class LatentKMeans(object):

    '''
    Takes a pretrained autoencoder model
    TODO: implement pretraining
    '''

    def __init__(self, model, nclusters, *args, **kwargs):
        self.n_clusters=n_clusters
        self.model = model
        self.kmeans = KMeans(nclusters, *args, **kwargs)

        hidden = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

    # follow the DCEC convention above and implement the following methods:

    def extract_feature(self, x):
        return self.encoder.predict(x)

    def fit(self, x):
        features = self.extract_feature(x)
        self.kmeans.fit(x)

    def predict(self, x):
        return self.kmeans.predict(x)

class VaDE(object):

    def __init__(self):
        raise NotImplementedError