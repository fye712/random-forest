import dill
from RandomTree import *

'''
This class implements a Random Forest classifier. It takes two arguments, 
number of trees (classifiers) in the ensemble and the number of features to be considered each time a splitting point is decided.
'''
class RandomForest:
    def __init__(self, ktrees, n_features):
        self.trees = []
        self.n_features = n_features
        self.ktrees = ktrees
        self.max_depth = n_features
        
    def make_trees(self, data):
        for i in range(self.ktrees):
            print("making tree", i)
            self.trees.append(RandomTree(data, self.n_features))
    
    def predict_row(self, row):
        all_predictions = []
        for tree in self.trees:
            all_predictions.append(tree.predict_row(row))
        return max(set(all_predictions), key=all_predictions.count)

    def predict(self, data):
        results = []
        for row in data:
            results.append(self.predict_row(row))
        return results

    def write_to_file(self, file):
        with open(file, 'wb') as f:
            dill.dump(self, f)

    