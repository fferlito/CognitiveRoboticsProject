import sys
import time

import numpy as np

from load_dataset import load_vfh_data, load_good5_data, load_good15_data
#from mondrian_forest_classifier_with_al_strategy import MondrianForestClassifierWithALStrategy
from online_kmean import OnlineKMeans
from test_ground import compute_purity

ini_amount_depth_data_thres = 300
ini_amount_depth_data_perc = 3000
num_trees_depth_data = 22

shape_descriptor = int(sys.argv[1])

if shape_descriptor == 0:
    data, labels, cv_generator = load_vfh_data()
elif shape_descriptor == 1:
    data, labels, cv_generator = load_good5_data()
elif shape_descriptor == 2:
    data, labels, cv_generator = load_good15_data()
else:
    print("undefined shape descriptor")

print(data.shape)
print(len(np.unique(labels)))

scores = []
samples_used = []

maxacc = 0
maxI = 0

for value in range(0, 50, 1):
    print(value)
    start_time = time.time()

    
    for training_idxs, validation_idxs in cv_generator:
        #mf = MondrianForestClassifierWithALStrategy(n_estimators=22)
        on = OnlineKMeans(75, 51, (value, 0))
        #mf.fit(data[training_idxs], labels[training_idxs])
        on.fit(data[training_idxs])
        #print('Fitted!')
        samples_used.append(len(labels[training_idxs]))
        #print(np.shape(data[validation_idxs, :]))
        test = on.predict(data[validation_idxs, :])
    # print(labels[validation_idxs])
        #print(len(validation_idxs))
        #print(len(validation_idxs))
        #print(len(np.unique(test)))
        #print(np.unique(test))
        scores.append(compute_purity(labels[validation_idxs],test , 51))
        #scores.append(mf.score(np.array(data[validation_idxs, :]), np.array(labels[validation_idxs])))
        #del mf
        print(scores)
        

    print("\n--------")
    #print("Performing cross-validation for descriptor {} data with with no AL".format(shape_descriptor))
    #function_time = time.time() - start_time
    #print("The cross validation took {} minutes and {} seconds".format(function_time // 60, function_time % 60))
    #print("--------")
    #print("Samples used: {:.2f} +- {:.2f}".format(np.mean(samples_used), np.std(samples_used)))
    #print(samples_used)
    #print("--------")
    print("Accuracies: {:.2f} +- {:.2f}".format(np.mean(scores) * 100, np.std(scores) * 100))
    print(scores)

    if np.mean(scores) > maxacc:
        maxacc = np.mean(scores)
        maxI = value


        
print(maxacc)
print(maxI)
    #print("--------\n")
