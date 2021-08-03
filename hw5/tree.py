#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:45:31 2021

@author: michail
"""

import numpy as np
from collections import namedtuple
from scipy import optimize


Leaf = namedtuple('Leaf', ('depth', 'y'))
Node = namedtuple('Node', ('depth', 'i', 'left', 'right', 'val'))

class Tree_base:
    def __init__(self, x, y, max_depth=45): #clss=0 --> node clss=1 --> leaf 
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.tree = self.make_tree_new(self.x, self.y)
        
    def crit2 (self, val, xi, y):
        N = len(y)
        y_l = y[xi <= val]
        y_r = y[xi > val]
        return (len(y_l)/N * np.std(y_l) + len(y_r)/N * np.std(y_r))

    def optimal_split(self, x, y):
        arg = []
        for i in range(x.shape[1]):
            ar = optimize.minimize_scalar(lambda ar: self.crit2(ar, x[:, i], y), bounds=(np.min(x[:,i]), np.max(x[:,i])),method='Bounded')
            arg.append(ar)
        i_ans = np.argmin(list(map(lambda x: x.fun, arg)))
        val = arg[i_ans].x
        return i_ans, val
    
    def make_tree_new(self, x, y, depth=1, min_samples=5):
        if (depth < self.max_depth) and (len(y) > min_samples):
            i, val = self.optimal_split(x, y)
            left = self.make_tree_new(x[x[:, i]<=val], y[x[:, i] <= val], depth=depth+1)
            right = self.make_tree_new(x[x[:, i]>val], y[x[:, i] > val], depth=depth+1)
            return Node(depth, i, left, right, val)
        else:
            return Leaf(depth, y)
    
    def predict(self, x):
        y = np.empty(x.shape[0], dtype=self.y.dtype)
        for t, cord in enumerate(x):
            node = self.tree
            while not isinstance(node, Leaf):
                i = node.i
                if cord[i] >= node.val:
                    node = node.right
                else:
                    node = node.left
            y[t] = node.y.mean()
        return y