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
                self.partial_fit([data], [label])
                self.partial_fit([shuffled_X[i]], [shuffled_Y[i]])
                n_samples_used += 2
                # train again on 2 * n_train instances (half of the unsure class, half random)
                #object_idxs = self.random_sample_from_one_instance(Y, label, n_train)
               # random_idxs = np.random.choice(len(Y), size=n_train)
               # print(object_idxs, random_idxs)
                # fit again for objects of the unsure instance with always a random one in between
               # for i in range(0, n_train):
               #     self.partial_fit([X[random_idxs][i]], [Y[random_idxs][i]])
                  #  n_samples_used += 2
        return n_samples_used

    def our_second_al_strategy(self, X, Y, classes = np.array(range(51)), inital_dataset_size = 300):
        classes = np.array(range(51)) if classes is None else classes  # default classes are 0-50
        shuffeled_X, shuffeled_Y = self.shuffling(X, Y)
        # fit to shuffled inital dataset
        self.partial_fit(shuffeled_X[:inital_dataset_size], shuffeled_Y[:inital_dataset_size], classes)
        correct_cnt = 0
        n_consecutive_correct_next_class = 50
        i = 0
        n_samples_used = 0
        skip_to_next = 0
        classes_done = 0
        while i < len(Y):
            pred = self.predict([X[i]])
            if pred == Y[i]:
                correct_cnt += 1
            else:
                correct_cnt = 0
            if correct_cnt == n_consecutive_correct_next_class:
                classes_done += 1
                if classes_done < 51:
                    next_instance_idx = np.where(Y == Y[i]+1)
                    dummy = np.sort(next_instance_idx)
                    x = dummy[0][0]
                    correct_cnt = 0
                    skip_to_next = 1
            if skip_to_next == 1:
                i = x
                skip_to_next = 0
            else:
                i += 1
            if classes_done == 51:
                print("Finished!")
                break
            if i > len(Y) - 1:
                print("Ran out of data, try smaller number of consecutive correctly predicted classes")
                break
            self.partial_fit([X[i]], [Y[i]], classes)
            self.partial_fit([shuffeled_X[i]], [shuffeled_Y[i]], classes)
            n_samples_used += 2
        return n_samples_used
    
class MondrianForest(ActiveLearning, MondrianForestClassifier):
    pass
    
class GaussianNaiveBayes(ActiveLearning, GaussianNB):
    pass
    
class knnClassifier(ActiveLearning, KNNClassifier):
    pass
    

