import sys
import warnings
import time
import numpy as np
from load_dataset import load_vfh_data, load_good5_data, load_good15_data
from mondrian_forest_classifier_with_al_strategy import *
import argparse

from skmultiflow.meta import AdaptiveRandomForestClassifier
from sklearn.naive_bayes import BernoulliNB



def train_and_test(args):
    ini_amount_depth_data_thres = 300
    ini_amount_depth_data_perc = 3000
    num_trees_depth_data = 22


    print('-------------------')
    print('Loading dataset...')
    if args.descriptor == 0:
        data, labels, cv_generator = load_vfh_data()
    elif args.descriptor == 1:
        data, labels, cv_generator = load_good5_data()
    elif args.descriptor == 2:
        data, labels, cv_generator = load_good15_data()
    else:
        print("undefined shape descriptor")
    
    print('Dataset dimensions:')
    print(data.shape)
    print('-------------------')

    scores = []
    samples_used = []

    print('Started training and testing...')
    print('-------------------')
    start_time = time.time()
    for training_idxs, validation_idxs in cv_generator:
        print('HIT ME BABY ONE MORE TIME')
        if args.classifier == 0:
            mf = KNNClassifierWithALStrategy(max_window_size = 1000,n_neighbors = args.k)
        elif args.classifier == 1:
            mf = BernoulliNBClassifierWithALStrategy()
        elif args.classifier == 2:
            mf = NaiveBayesClassifierWithALStrategy()
        else:
            print('ERROR: pleaase specify a valid classifier (i.e. --classifier 0)')
            print('Quitting the code..')
            
        if args.strategy == 0:
            samples_used.append(mf.fit_using_al_strategy_thres_intermediate_update(data[training_idxs], labels[training_idxs],
                                                                        np.array(range(51)), 300, args.threshold))
        elif args.strategy == 1:
            samples_used.append(mf.fit_using_al_strategy_thres(data[training_idxs], labels[training_idxs],
                                                                        np.array(range(51)), 300, args.threshold))
        elif args.strategy == 2:
            samples_used.append(mf.our_al_strategy(data[training_idxs], labels[training_idxs]))
        elif args.strategy == 3:
            samples_used.append(mf.our_second_al_strategy(data[training_idxs], labels[training_idxs]))
        else:
            print('ERROR: pleaase specify a valid strategy (i.e. --strategy 0)')
            print('Quitting the code..')
                
        scores.append(mf.score(np.array(data[validation_idxs, :]), np.array(labels[validation_idxs])))
        
        del mf

    print("\n--------")
    print("Performed cross-validation using a threshold of {}".format(args.threshold))
    function_time = time.time() - start_time
    print("The cross validation took {} minutes and {} seconds".format(function_time // 60, function_time % 60))
    print("--------")
    print("Samples used: {:.2f} +- {:.2f}".format(np.mean(samples_used), np.std(samples_used)))
    print(samples_used)
    print("--------")
    print("Accuracies: {:.2f} +- {:.2f}".format(np.mean(scores) * 100, np.std(scores) * 100))
    print(scores)
    print("--------\n")


def main():
    parser = argparse.ArgumentParser(description="Cognitive robotics project (group 7)", formatter_class=argparse.RawTextHelpFormatter)
    # Set the dataset parser
    parser.add_argument('--descriptor', type=int, required=True, help= '''Dataset to use:   
   0 -> VFH 
   1 -> GOOD 5
   2 -> GOOD 15
              ''')
    # Set the strategy parser
    parser.add_argument('--strategy', type=int, required=True, help= '''Active learning strategy to use: 
   0 -> fit_using_al_strategy_thres_intermediate_update
   1 -> fit_using_al_strategy_thres
   2 -> our_al_strategy
   3 -> our_second_al_strategy
''')
    # Set the threshold parser
    parser.add_argument('--threshold', type=float, help='''Threhsold in the range 0-1 to use for the active learning strategy''')
    # Set the strategy parser
    parser.add_argument('--classifier', type=int, required=True, help= '''Classifier to use: 
   0 -> KNNclassifier
   1 -> BernoulliNB
   2 -> MLPClassifierClassifierWithALStrategy
''')
    parser.add_argument('--k', type=int, help= '''Number of neighbors for KNN (i.e. 1) 
              ''')

    # parse inputs
    args = parser.parse_args()
    
    
    
    if args.strategy == 0 and args.threshold==None:
        print('ERROR: You need to specify a threshold (i.e. --threshold 0.5) using this strategy')
        print('Quitting the code..')
        quit()

    if args.classifier == 0 and args.k==None:
        print('ERROR: You need to specify a a variable')
        print('Quitting the code..')
        quit()
        
    print('\nCognitive robotics project (group 7)\n')
    # RUN MAIN PART
    train_and_test(args)
    

    
if __name__ == "__main__":
    main()

