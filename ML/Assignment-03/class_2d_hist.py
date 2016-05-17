#!/usr/bin/env python
import numpy as np

class histogram_2d:
    def __init__(self, two_d_training_set, min_d1, max_d1, min_d2, max_d2, bin_size) :
        self.training_set = two_d_training_set
        self.min_d1 = min_d1
        self.max_d1 = max_d1
        self.min_d2 = min_d2
        self.max_d2 = max_d2
        self.bin_size = bin_size

    def get_bin_index(self, item, min_range, max_range):
        return (float (1 + ((self.bin_size -1)*((item - min_range)/(max_range-min_range)))))

    def get_2d_histogram(self):
        histogram_2d = np.zeros([self.bin_size, self.bin_size],dtype=float)
        for d1,d2 in self.training_set:
            r = self.get_bin_index(d1, self.min_d1, self.max_d1)
            c = self.get_bin_index(d2, self.min_d2, self.max_d2)
            histogram_2d[r-1, c-1] += 1
        self.histogram_2d = histogram_2d
        return histogram_2d

    def  get_bin_count(self, p1, p2):
        r = round(self.get_bin_index(p1, self.min_d1, self.max_d1))
        c = round(self.get_bin_index(p2, self.min_d2, self.max_d2))
        bc = self.histogram_2d[r-1][c-1]
        return bc

