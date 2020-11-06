import numpy as np
from skgarden import MondrianForestClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from skmultiflow.data import DataStream
from sklearn.linear_model import Perceptron
from skmultiflow.meta.multi_output_learner import MultiOutputLearner
class MondrianForestClassifierWithALStrategy(AdaptiveRandomForestClassifier):
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

    def fit_using_al_strategy_thres(self, X, Y, classes=None, inital_dataset_size=300, threshold=0.5):
        """
        This method trains on the initial data and then selects the rest of the training samples based on the
        threshold and than learns on all selected samples
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

        # Retrieve the probability distributions for the remaining training samples
        probabilities = self.predict_proba(X[inital_dataset_size:])
        # Filter the training samples based on the confidence of the mf
        selected_idxs = [idx for probs, idx in zip(probabilities, range(inital_dataset_size, X.shape[0])) if
                            self.calculate_confidence(probs) < threshold]
        # Fit again on the samples the mf is unsure about
       # self.partial_fit(X[selected_idxs], Y[selected_idxs])

        return len(selected_idxs)

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

    def random_sample_from_one_instance(self, Y, instance):
        instance_idxs = np.where(Y == instance)
        random_sample_of_instance = np.random.choice(instance_idxs[0], size=25)
        return random_sample_of_instance

    def our_al_strategy(self, X, Y, classes = np.array(range(51)), inital_dataset_size=300, threshold = 0.5, n_of_objects_confidence = 51):
        classes = np.array(range(51))
        train_X, train_Y = self.shuffling(X, Y)
        # fit to shuffled inital dataset
        self.partial_fit(train_X[:inital_dataset_size], train_Y[:inital_dataset_size], classes)
        for data, label in zip(X, Y):
            # Calculate confidence in the prediction for the data sample
            if self.calculate_confidence(self.predict_proba([data])[0]) < threshold:
                # train on exact instance that is was unsure about
                self.partial_fit([data], [label])
                # train again on 50 instances (25 of the unsure class, 25 random)
                object_idxs = self.random_sample_from_one_instance(Y, label)
                random_idxs = np.random.choice(Y, size=25)
                # fit again for objects of the unsure instance with always a random one in between
                for i in range(0, 25):
                    self.partial_fit([X[object_idxs][i]], [Y[object_idxs][i]])
                    self.partial_fit([X[random_idxs][i]], [Y[random_idxs][i]])
        return 1

    def our_second_al_strategy(self, X, Y, classes = np.array(range(51)), inital_dataset_size = 300, threshold=0.5):
        train_X, train_Y = self.shuffling(X, Y)
        # fit to shuffled inital dataset
        self.partial_fit(train_X[:inital_dataset_size], train_Y[:inital_dataset_size], classes)
        correct_cnt = 0
        n_cor_continue = 5
        cnt = 0
        print("length: " + str(len(Y)))
        i = 0
        skip_to_next = 0
        classes_done = 0
        while i < len(Y):
            print(i)
            pred = self.predict([X[i]])
            if pred == Y[i]:
                correct_cnt += 1
                print("correct YAY")
            if correct_cnt == n_cor_continue:
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
            print("i after: " + str(i))
            cnt += 1
            print("loop executed: " + str(cnt))
            self.partial_fit([X[i]], [Y[i]])
            self.partial_fit([train_X[i]], [train_Y[i]])
        return 1

