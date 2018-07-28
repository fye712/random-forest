import numpy as np
from DecisionTree import Node, binary_split, gini_index
'''
This method finds the best splitting point from n features out of the total number of features.
'''
def get_random_split(data, is_categorical, n_features):
    # go over all attributes
    classes = list(set(row[-1] for row in data))
    ind, split, best_gini, best_groups = 0, 0, 1, []
    num_features = len(data[0]) - 1
    sub = np.random.choice(range(num_features), n_features, replace=False)
    for att in sub:
        # print('checking ', att)
        att_vals = list(set(row[att] for row in data))
        mini = min(att_vals)
        maxi = max(att_vals)
        if (mini == maxi):
            continue
        if att_vals == [0, 1]:
            for i in [0, 1]:
                groups = binary_split(att, i, data)
                gini = gini_index(groups, classes)
                if gini < best_gini:
                    ind, split, best_gini, best_groups = att, i, gini, groups
            continue
        for i in np.arange(mini, maxi, (maxi - mini) / 100):
            # print(i)
            if att in is_categorical:
                groups = categorical_split(att)
            else:
                groups = binary_split(att, i, data)
            gini = gini_index(groups, classes)
            if gini < best_gini:
                ind, split, best_gini, best_groups = att, i, gini, groups
    return Node(ind, split, best_groups) # might just need attribute and splitting value

'''
This class implements a decision tree which uses the random split algorithm for use in Random Forest.
'''
class RandomTree:
    def __init__(self, data, n_features):
        self.n_features = n_features
        self.max_depth = 8
        self.root = self.build_tree(data, 0)

    
    def build_tree(self, data, depth):
        # check all the classes
        targets = list(row[-1] for row in data)
        if len(set(targets)) == 1 or depth >= self.max_depth or len(targets) == 0: # pure leaf
            print(depth)
            clas = max(set(targets), key=targets.count)
            return Node(0, 0, [], clas)
        root = get_random_split(data, [], self.n_features)
        for children in root.training_groups: # data in the children of the root
            child_node = self.build_tree(children, depth + 1)
            root.add_child(child_node)
        return root

    def predict_row(self, row):
        curnode = self.root
        while (curnode.clas is None):

            if row[curnode.index] < curnode.value:
                curnode = curnode.children[0] # go to left child
            else:
                curnode = curnode.children[1]
        return curnode.clas
    
    def predict(self, data):
        results = []
        for row in data:
            results.append(self.predict_row(row))
        return results