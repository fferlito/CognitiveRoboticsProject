# Cognitive Robotics Project 2020
Project for group number 7

Different different classifiers (KNN, GaussianNaiveBaye and Mondrian Forest Classifier) are trained  on feature histograms computed using the Global Orthographic Object Descriptor (GOOD) object descriptors and on the VFH descriptor. The training is done using three different active learning strategies.

We use the Washington RGB-D dataset, available here:

 https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/ 
 
 https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/


To run the code: 
1. first compile the c++ code in the CPP folder. Please refer to https://github.com/SeyedHamidreza/GOOD_descriptor
2. download the dataset 
3. Install the python modules, specified in the requirements.txt file. We suggest to install the packages one by one, using the versions indicated.
4. Build the dataset (build_additional_dataset.py and build_dataset.py) changing the path variables EVAL_DATASET_PATH, PC_DATASET_PATH, OUTPUT_DATASET_PATH
5. Run either the train.py to generate the run the cross-validation and get the results. The scripts requires the following flags:
You need to include the following flags: 
``` 
--descriptor 0 -> VFH 
             2 -> GOOD 15
                                      
--strategy   0 -> fit_using_al_strategy_thres_intermediate_update
             1 -> our_al_strategy
             2 -> our_second_al_strategy
                           
--threshold (in the range 0-1 to use for the active learning strategy 0 and 1)

--classifier 0 -> Mondrian Forest Classifier
             1 -> Gaussian Naive Bayes Classifier
             2 -> KNN
           
 --k integer (k neighbors when using KNN as classifier)

```
