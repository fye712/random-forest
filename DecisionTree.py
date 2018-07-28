import numpy as np
'''
This calculates the gini index.
'''
def gini_index(groups, classes): # groups is a list of sets of things resulting from the split, classes is target classes
    num_data = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        sub_gini = 1
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            sub_gini -= p * p
        gini += sub_gini * size / num_data
    return gini

'''
This method finds the best split from training data.
'''
def get_split(data, is_categorical):
    # go over all attributes
    classes = list(set(row[-1] for row in data))
    ind, split, best_gini, best_groups = 0, 0, 1, []
    num_features = len(data[0]) - 1
    for att in range(num_features):
        for row in data:
            if att in is_categorical:
                groups = categorical_split(att)
            else:
                groups = binary_split(att, row[att], data)
            gini = gini_index(groups, classes)
            if gini < best_gini:
                ind, split, best_gini, best_groups = att, row[att], gini, groups
                print (len(groups[0]), len(groups[1]), best_gini)
    return Node(ind, split, best_groups) # might just need attribute and splitting value
                
    
'''
This method separates a dataset into two groups based on a splitting value.
'''
def binary_split(index, value, data):
    left, right = list(), list()
    for row in data:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

'''
This method separates a data set by its categorical values.
'''
def categorical_split(index, values):
    leafs = {}
    for value in values:
        leafs[value] = list()
    for row in data:
        leafs[row[index]].append(row)
    return leafs

'''
This class represents a node in the decision tree.
'''
class Node:
    def __init__(self, index, value, groups, clas=None):
        self.index = index
        self.value = value
        self.children = []
        self.training_groups = groups
        self.clas = clas
        
    def add_child(self, node):
        self.children.append(node)

'''
This class implements a decision tree which builds until each leaf is pure.
'''
class DecisionTree:
    def __init__(self, data):
        self.root = self.build_tree(data)
    
    def build_tree(self, data):
        # check all the classes
        targets = set(row[-1] for row in data)
        if len(targets) == 1: # pure leaf
            return Node(0, 0, [], list(targets)[0])
        root = get_split(data, [])
        for children in root.training_groups: # data in the children of the root
            child_node = self.build_tree(children)
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