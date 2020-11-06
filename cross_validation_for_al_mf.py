
import sys
import warnings
import time
import numpy as np
from load_dataset import load_vfh_data, load_good5_data, load_good15_data
from mondrian_forest_classifier_with_al_strategy import MondrianForestClassifierWithALStrategy
import argparse



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
        mf = MondrianForestClassifierWithALStrategy(n_estimators=22)
       
        if args.classifier == 0:
            samples_used.append(mf.fit_using_al_strategy_thres_intermediate_update(data[training_idxs], labels[training_idxs],
                                                                        np.array(range(51)), 300, args.threshold))
        elif args.classifier == 1:
            samples_used.append(mf.fit_using_al_strategy_thres_intermediate_update(data[training_idxs], labels[training_idxs],
                                                                        np.array(range(51)), 300, args.threshold))
        elif args.classifier == 1:
            samples_used.append(mf.fit_using_al_strategy_thres_intermediate_update(data[training_idxs], labels[training_idxs],
                                                                        np.array(range(51)), 300, args.threshold))
        else:
            print('ERROR: pleaase specify a valid classifier (i.e. --classifier 0)')
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
   0 -> threhsold intermediate update
   1 -> new strategy
''')
    # Set the threshold parser
    parser.add_argument('--threshold', type=float, help='''Threhsold in the range 0-1 to use for the active learning strategy''')
    # Set the strategy parser
    parser.add_argument('--classifier', type=int, required=True, help= '''Classifier to use: 
   0 -> Mondrian forest
   1 -> BernoulliNB??
   2 -> Generic classifier??
''')
    # parse inputs
    args = parser.parse_args()
    
    
    
    if args.strategy == 0 and args.threshold==None:
        print('ERROR: You need to specify a threhsold (i.e. --threshold 0.5) using this strategy')
        print('Quitting the code..')
        quit()
        
    print('\nCognitive robotics project (group 7)\n')
    # RUN MAIN PART
    train_and_test(args)
    

    
if __name__ == "__main__":
    main()
