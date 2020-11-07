# Cognitive Robotics Project 2020
Project for group number 7

based on: 

https://nestor.rug.nl/bbcswebdav/pid-10289157-dt-content-rid-26510189_2/courses/WMAI003-05.2020-2021.1A/example_final_project.pdf


In this project we consider training a classifier X on feature histograms computed using the Global Orthographic Object Descriptor (GOOD) object descriptors and another descriptor X. 

We trained the classificator both using active and offline learning. When training with active learning uncertainty sampling was used as a querying strategy.
We use the Washington RGB-D dataset and compared the results we found to results found in other papers using the same dataset. 


To run the code: 
1. first compile the c++ code in the CPP folder. Please refer to https://github.com/SeyedHamidreza/GOOD_descriptor
2. download the dataset from https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/ and https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/
3. Install the python modules, specified in the requirements.txt file:
``` 
pip install -r requirements
``` 

4. Build the dataset (build_additional_dataset.py) changing the path variables EVAL_DATASET_PATH, PC_DATASET_PATH, OUTPUT_DATASET_PATH
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
