import sys
import warnings
import time
import numpy as np
from load_dataset import load_vfh_data, load_good15_data
from ActiveLearning import *
import argparse
import warnings
import gc


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def train_and_test(args):
    print('-------------------')
    print('Loading dataset...')
    if args.descriptor == 0:
        data, labels, cv_generator = load_vfh_data()
    elif args.descriptor == 1:
        data, labels, cv_generator = load_good15_data()
    else:
        print("undefined shape descriptor")
    
    print('Dataset dimensions:')
    print(data.shape)
    print('-------------------')

    scores = []
    samples_used = []

    print('Started training using cross validation...')
    print('-------------------')
    start_time = time.time()
    stopper = 0
    for training_idxs, validation_idxs in cv_generator:
        if args.classifier == 0:
            mf = MondrianForest(n_estimators=10, max_depth =1000)
        elif args.classifier == 1:
            mf = GaussianNaiveBayes(var_smoothing = 0.001)
        elif args.classifier == 2:
            mf = knnClassifier()
        else:
            print('ERROR: pleaase specify a valid classifier (i.e. --classifier 0)')
            print('Quitting the code..')
            
        if args.strategy == 0:
            samples_used.append(mf.fit_using_al_strategy_thres_intermediate_update(data[training_idxs], labels[training_idxs],np.array(range(51)), inital_dataset_size=1000, threshold=args.threshold))
        elif args.strategy == 1:
            samples_used.append(mf.our_al_strategy(data[training_idxs], labels[training_idxs], np.array(range(51)), 1000, args.threshold))
        elif args.strategy == 2:
            samples_used.append(mf.second_al_strategy(data[training_idxs], labels[training_idxs], np.array(range(51)), 1000))
        else:
            print('ERROR: pleaase specify a valid strategy (i.e. --strategy 0)')
            print('Quitting the code..')
        print('Finished fold..')
                
        scores.append(mf.score(np.array(data[validation_idxs, :]), np.array(labels[validation_idxs])))
        stopper += 1
        if stopper == 1:
            break
        del mf
        gc.collect()


    print("\n--------")
    print("Performed cross-validation")
    function_time = time.time() - start_time
    print("The cross validation took {} minutes and {} seconds".format(function_time // 60, function_time % 60))
    print("--------")
    print("Samples used: {:.2f} +- {:.2f}".format(np.mean(samples_used), np.std(samples_used)))
    print(samples_used)
    print("--------")
    print("Accuracies: {:.2f} +- {:.2f}".format(np.mean(scores) * 100, np.std(scores) * 100))
    print(scores)
    print("--------\n")
    print("The cross validation took {} minutes and {} seconds".format(function_time // 60, function_time % 60))
    print("--------\n")


def main():
    parser = argparse.ArgumentParser(description="Cognitive robotics project (group 7)", formatter_class=argparse.RawTextHelpFormatter)
    # Set the dataset parser
    parser.add_argument('--descriptor', type=int, required=True, help= '''Dataset to use:   
   0 -> VFH 
   1 -> GOOD 15''')
    
    # Set the strategy parser
    parser.add_argument('--strategy', type=int, required=True, help= '''Active learning strategy to use: 
   0 -> fit_using_al_strategy_thres_intermediate_update
   1 -> our_al_strategy
   2 -> our_second_al_strategy''')
    
    # Set the threshold parser
    parser.add_argument('--threshold', type=float, help='''Threhsold in the range 0-1 to use for the active learning strategy (strategy: 0 and 1)''')
    
    # Set the strategy parser
    parser.add_argument('--classifier', type=int, required=True, help= '''Classifier to use: 
   0 -> Mondrian Forest Classifier
   1 -> Gaussian Naive Bayes Classifier
   2 -> KNN
''')
    parser.add_argument('--k', type=int, help= '''Number of neighbors for KNN (i.e. 1)''')

    # parse inputs
    args = parser.parse_args()
    
    
    
    if (args.strategy == 0 or args.strategy == 1) and args.threshold==None:
        print('ERROR: You need to specify a threshold (i.e. --threshold 0.5) using this strategy')
        print('Quitting the code..')
        quit()
        
    if args.classifier == 2 and args.k==None:
        print('ERROR: You need to specify a number of neighbor k for knn (i.e. --k 3)')
        print('Quitting the code..')
        quit()
        
    print('\nCognitive robotics project (group 7)\n')
    # RUN MAIN PART
    train_and_test(args)
    
 
    
 

    

    
if __name__ == "__main__":
    main()

