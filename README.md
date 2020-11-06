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
3. Build the dataset (build_additional_dataset.py) changing the path variables EVAL_DATASET_PATH, PC_DATASET_PATH, OUTPUT_DATASET_PATH
4. Run either the cross_validation_for_al_mf.py shape_descriptor confidence_threshold(shape descriptor should be 0(VFH), 1(GOOD5) or 2(GOOD15)) or cross_validation_for_non_al_mf.py to generate the results. The required flags are:
You need to include the following flags: 
``` 
--descriptor Dataset to use:   
                           0 -> VFH 
                           1 -> GOOD 5
                           2 -> GOOD 15
                                      
--strategy  Active learning strategy to use: 
                           0 -> threhsold intermediate update
                           1 -> new strategy
                           
--threshold Threhsold in the range 0-1 to use for the active learning strategy

--classifier Classifier to use: 
                           0 -> Mondrian forest
                           1 -> BernoulliNB??
                           2 -> Generic classifier??
```
