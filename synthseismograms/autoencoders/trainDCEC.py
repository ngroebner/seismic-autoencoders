from autoencoders import *
import argparse


if __name__ == 'main':
    
    parser = argparse.ArgumentParser(description='train')
    #parser.add_argument('dataset', default='mnist', choices=['mnist', 'usps', 'mnist-test'])
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e5, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results/temp')
    args = parser.parse_args()

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load the data here
    
    