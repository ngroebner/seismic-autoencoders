import os
import argparse


parser = argparse.ArgumentParser(description='Run multiple fits for Experiment 2, for all of the algorithm types.')
parser.add_argument('nruns', type=int, 
                    help='number of runs performed for each algorithm')


args = parser.parse_args()

nruns = args.nruns

if __name__ == '__main__':
    
    '''for i in range(nruns+1):
        print('Run {} of {}'.format(i+1, nruns+1))
        os.system('python .\Experiment2.py with causal_skips')'''

    for i in range(nruns+1):
        print('Run {} of {}'.format(i+1, nruns))
        os.system('python .\Experiment2.py with causal_noskips')

    for i in range(nruns+1):
        print('Run {} of {}'.format(i+1, nruns))
        os.system('python .\Experiment2.py with acausal_skips')

    for i in range(nruns+1):
        print('Run {} of {}'.format(i+1, nruns))
        os.system('python .\Experiment2.py with acausal_noskips')
