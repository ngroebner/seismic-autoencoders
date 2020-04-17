import numpy as np
from autoencoders import  DCEC, CausalCAE, CAE
import os
from sklearn.metrics import silhouette_score
from keras.utils import multi_gpu_model
import tensorflow as tf
import pickle

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('Experiment3')
ex.observers.append(MongoObserver(url='127.0.0.1:27017',
                                  db_name='seismicautoencoders'))
ex_dir = 'Experiment3/'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@ex.config
def configuration():
    data_dir = 'Experiment3'   
    n_clusters = 2
    nnet_type = '1D'
    algorithm = 'causal'
    skips = True
    n_filters = 64
    latent_dims = 10
    n_layers = 4
    lamda = 0.9
    tol = 1e-8
    maxiter = 1e5
    save_dir = ''
    update_interval=100
    batch_size = 256
    
@ex.named_config
def causal_skips():
    algorithm = 'causal'
    skips = True
    
@ex.named_config
def causal_noskips():
    algorithm = 'causal'
    skips = False
    
@ex.named_config
def acausal_skips():
    algorithm = 'acausal'
    skips = True
    
@ex.named_config
def acausal_noskips():
    algorithm = 'acausal'
    skips = False
    

@ex.automain
def main(_run, data_dir, algorithm, skips, nnet_type,
         n_clusters, 
         n_filters, latent_dims, n_layers, 
         lamda, tol, maxiter, save_dir, 
         update_interval, batch_size):
    
    with open(data_dir+'/synthetics.pkl', 'rb') as f:
        synthetics = pickle.load(f)

    X = synthetics['spectrograms']
    #convert source_mechanisms string into int variable for accuracy calculations
    # I might want to include this in the data package, or perhaps class the data package
    noise2cat = {'GR':0, 'Gaussian':1}
    
    y = np.array([noise2cat[noise] for noise in synthetics['noise_types']])

    save_dir = ex_dir + save_dir + '/{}_skips{}'.format(algorithm,skips) 
    print("Saving into {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    input_shape = (X[0].shape[0], X[0].shape[1])
    
    if algorithm == 'causal': 
        model = CausalCAE(nnet_type=nnet_type, latent_dims=latent_dims, input_shape=input_shape, 
                          n_filters=n_filters, n_layers=n_layers, 
                          use_skip_connections=skips)
    elif algorithm == 'acausal':
        model = CAE(nnet_type=nnet_type, latent_dims=latent_dims, input_shape=input_shape, 
                          n_filters=n_filters, n_layers=n_layers, 
                          use_skip_connections=skips)


    silhouette_scores = [] 


    save_dir_n = save_dir + '/' + str(n_clusters) + 'clusters'
    if not os.path.exists(save_dir_n):
            os.makedirs(save_dir_n)

    dcec = DCEC(model=model,
                        latent_dims=latent_dims,
                        input_shape=input_shape,
                        n_filters=n_filters, n_layers=n_layers,
                        n_clusters=n_clusters,
                        lamda=lamda,
                        multi_gpu=False)

    dcec.compile()  
    #print('Loading pretrained model weights...')
    #dcec.cae.load_weights('{}/pretrain_cae_model.h5'.format(save_dir))
    #dcec.pretrained = True 
    print('*************** Fitting model with {} clusters ***************'.format(n_clusters))
    [history, pretrain_history] = dcec.fit(X, y=y, tol=tol, maxiter=maxiter,
                           batch_size=batch_size,
                           update_interval=update_interval,
                           save_interval=100,
                           save_dir=save_dir_n)

    # suppress printing?

    print('Predicting clusters...')
    clusters = dcec.predict(X)
    cluster_filename = '{}/cluster_assignments_{}_clusters.npy'.format(save_dir_n,n_clusters)
    np.save(cluster_filename,clusters)
    ex.add_artifact(cluster_filename)

    latent_vectors = dcec.extract_feature(X)
    lv_filename = '{}/latent_vectors_{}_clusters.npy'.format(save_dir_n,n_clusters)
    np.save(lv_filename,latent_vectors)
    ex.add_artifact(lv_filename)

    score = silhouette_score(latent_vectors, clusters)
    _run.log_scalar('silhouette score', score, n_clusters)
    silhouette_scores.append(score)

    #log the losses and acuracy metrics
    for ite,L in enumerate(history['L']):
        _run.log_scalar('total loss_{}'.format(n_clusters), L, ite)
    for ite,Lc in enumerate(history['Lc']):
        _run.log_scalar('Lc(clustering loss_{}'.format(n_clusters), Lc, ite)
    for ite,Lr in enumerate(history['Lr']):
        _run.log_scalar('Lr(reconstruction loss_{}'.format(n_clusters), Lr, ite)  
    for ite,acc in enumerate(history['acc']):
        _run.log_scalar('accuracy_{}'.format(n_clusters), acc, ite)
    for ite,nmi in enumerate(history['nmi']):
        _run.log_scalar('NMI_{}'.format(n_clusters), nmi, ite)
    for ite,ari in enumerate(history['ari']):
        _run.log_scalar('ARI_{}'.format(n_clusters), ari, ite)
    for ite,loss in enumerate(pretrain_history.history['loss']):
        _run.log_scalar('pretrain_loss_{}'.format(n_clusters), loss, ite)

    # final model weights and pretraining log
    ex.add_artifact((save_dir_n+'/dcec_model_final.h5'))
    ex.add_artifact(save_dir_n+'/pretrain_log.csv')
    ex.add_artifact(save_dir_n+'/pretrain_cae_model.h5')
    ex.add_artifact(cluster_filename)
    ex.add_artifact(lv_filename)

    print('Silhouette scores for number of clusters:')
    print('N Clusters::Silhouette score')
    
    print('{}::{}'.format(n_clusters, score))


