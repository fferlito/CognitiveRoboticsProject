import numpy as np

from skgarden import MondrianForestClassifier
from sklearn.naive_bayes import GaussianNB
from skmultiflow.lazy import KNNClassifier


class ActiveLearning():
    @staticmethod
    def calculate_confidence(probabilities):
        """
        :param probabilities:
        :return confidence:
        """
        sorted_proba = sorted(probabilities, reverse=True)
        return 1 - sorted_proba[1] / sorted_proba[0]

    @staticmethod
    def shuffling(X_train, y_train):
        shuffled_rows = np.random.permutation(X_train.shape[0])
        return X_train[shuffled_rows], y_train[shuffled_rows]

    @staticmethod
    def random_sampling(X_train, y_train, num_of_dataset):
        rand_selected_rows = np.random.choice(X_train.shape[0], num_of_dataset, replace=False)
        return X_train[rand_selected_rows, :], y_train[rand_selected_rows], rand_selected_rows

    def fit_using_al_strategy_thres_intermediate_update(self, X, Y, classes=None, inital_dataset_size=300,
                                                        threshold=0.5):
        """
        This method trains on the initial data and goes through the remaining data learning from the samples or not
        according to the confidence threshold
        :param X: Data samples
        :param Y: Data labels
        :param classes: all classes
        :param inital_dataset_size: the size of the initial training set
        :param threshold: the confidence threshold
        :return:
        """
        classes = np.array(range(51)) if classes is None else classes  # default classes are 0-50
        # first shuffle the dataset and get the initial data
        X, Y = self.shuffling(X, Y)
        # Partial fit on the initial data

        self.partial_fit(X[:inital_dataset_size], Y[:inital_dataset_size], classes)

        amount_of_training_samples_selected = 0
        # Loop over remaining data samples
        for data, label in zip(X[inital_dataset_size:], Y[inital_dataset_size:]):
            # Calculate confidence in the prediction for the data sample
            if self.calculate_confidence(self.predict_proba([data])[0]) < threshold:
                # Update the model if we are insecure
                self.partial_fit([data], [label])
                amount_of_training_samples_selected += 1

            
        return amount_of_training_samples_selected

    def random_sample_from_one_instance(self, Y, instance, n_train):
        instance_idxs = np.where(Y == instance)
        random_sample_of_instance = np.random.choice(instance_idxs[0], size=n_train)
        return random_sample_of_instance

    def our_al_strategy(self, X, Y, classes=np.array(range(51)), inital_dataset_size=300, threshold = 0.5):
        classes = np.array(range(51)) if classes is None else classes  # default classes are 0-50
        shuffled_X, shuffled_Y = self.shuffling(X, Y)
        n_train = 1 # change number of training instances here
        n_samples_used = 0
        # fit to shuffled inital dataset
        self.partial_fit(shuffled_X[:inital_dataset_size], shuffled_Y[:inital_dataset_size], classes)
        i = 0
        for data, label in zip(X, Y):
            i += 1
            # Calculate confidence in the prediction for the data sample
            if self.calculate_confidence(self.predict_proba([data])[0]) < threshold:
                # train on exact instance that is was unsure about
                if i >= len(Y):
                    i = 1
                self.partial_fit([data], [label])
                self.partial_fit([shuffled_X[i]], [shuffled_Y[i]])
                n_samples_used += 2
        print('Finished')
        return n_samples_used

    
    def second_al_strategy(self, X, Y, classes = np.array(range(51)), inital_dataset_size = 300):
        
        #classes = np.array(range(51)) if classes is None else classes  # default classes are 0-50
        shuffeled_X, shuffeled_Y = self.shuffling(X, Y)
        # fit to shuffled inital dataset
        self.partial_fit(shuffeled_X[:inital_dataset_size], shuffeled_Y[:inital_dataset_size], classes)
        
        training_samples_used = 0
        
        for type_of_object in classes:
            print(type_of_object)
            
            # get current class to train
            idx = np.argwhere(Y == type_of_object).flatten()
            current_objects = X[idx]
            current_labels = Y[idx]
            
            # get counter example
            mask = np.ones(Y.size, dtype=bool)
            mask[idx] = False
            other_labels = Y[mask]
            other_objects = X[mask]
            
            consecute_guesses = 0
            while consecute_guesses < 50:
                i = np.random.choice(len(current_objects))
                pred = self.predict([current_objects[i]])
                if pred == type_of_object:
                    consecute_guesses += 1
                else:
                    consecute_guesses = 0
                    counter_i = np.random.choice(len(other_objects))
                    self.partial_fit([current_objects[i]], [current_labels[i]], classes)
                    self.partial_fit([other_objects[counter_i]], [other_labels[counter_i]], classes)
                    training_samples_used += 2
                    
        return training_samples_used
            
            
class MondrianForest(ActiveLearning, MondrianForestClassifier):
    pass
    
class GaussianNaiveBayes(ActiveLearning, GaussianNB):
    pass
    
class knnClassifier(ActiveLearning, KNNClassifier):
    pass
    

