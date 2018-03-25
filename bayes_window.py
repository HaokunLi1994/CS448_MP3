#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 00:31:18 2018

@author: Haokun Li
"""

import itertools
import numpy as np

class Window(object):
    def __init__(self, height, width, overlap):
        """
        Args:
            height(int)
            width(int)
            overlap(boolean)
        Returns:
            (None)
        """
        self.height = height
        self.width = width
        self.overlap = overlap
        self.possible_values = []
        self.value_dict = {}
        
        # Generate all possible values
        number = self.height * self.width
        count_value = 0
        for value in itertools.product(('0', '1'), repeat=number):
            mystr = ''.join(value)
            self.value_dict.update({mystr: count_value})
            self.possible_values.append(count_value)
            count_value += 1
        pass

    def transform(self, X):
        """
        Use our window to transform training data.
        
        Args:
            X(np.array): training data
        Returns:
            new_X(np.array)
        """
        N, pic_height, pic_width = X.shape
        
        if self.overlap == False:
            new_height = int(pic_height / self.height)
            new_width = int(pic_width / self.width)
            new_X = np.zeros((N, new_height, new_width), dtype=np.int32)
            for count in range(N):
                pic = X[count]
                i = 0
                new_i = 0
                while (i + self.height) <= pic_height:
                    j = 0
                    new_j = 0
                    while (j + self.width) <= pic_width:
                        myslice = pic[i:i+self.height, j:j+self.width]
                        mystr = ''.join(''.join('%s' % x for x in y) for y in myslice)
                        new_X[count][new_i][new_j] = self.value_dict[mystr]
                        j += self.width
                        new_j += 1
                    i += self.height
                    new_i += 1
        else:
            # Overlap == True
            new_height = int(pic_height - self.height + 1)
            new_width = int(pic_width - self.width  + 1)
            new_X = np.zeros((N, new_height, new_width), dtype=np.int32)
            for count in range(N):
                pic = X[count]
                i = 0
                while (i + self.height) <= pic_height:
                    j = 0
                    while (j + self.width) <= pic_width:
                        myslice = pic[i:i+self.height, j:j+self.width]
                        mystr = ''.join(''.join('%s' % x for x in y) for y in myslice)
                        new_X[count][i][j] = self.value_dict[mystr]
                        j += self.width
                    i += self.height
                    
        return new_X
