"""
Sample Gaussian Filter for a casual system. Based on equation (1a) from article.

"""
from collections import deque
import numpy as np
import math

class StaticFilter:
    def __init__(self, sigma, N=None):
        self.sigma = float(sigma) 
        self.N0 = int(N) if N else int(5*sigma) #Approximation for one dimension
        self.clear_data()
        self.calculate_g_vec()

    def clear_data(self):
        self.in_vec = deque(maxlen=self.N0)

    def calculate_g_vec(self):
        temp = (math.sqrt(2*math.pi)*self.sigma)**-1
        self.g_vec = [temp]
        for i in range(1, self.N0):
            self.g_vec.append(temp * math.exp(-0.5*i/(self.sigma**2)))

    def restart(self):
        self.clear_data()

    def out_value(self, new_in_value):
        self.put(new_in_value)
        return self.conv_data()

    def put(self, new_in_value):
        self.in_vec.appendleft(new_in_value)

    def conv_data(self):
        sum_value = 0.
        sum_gk = 0.
        for in_k, g_k in zip(self.in_vec, self.g_vec):
            sum_value += in_k * g_k
            sum_gk += g_k
        return sum_value / sum_gk #Normalizing by incomplete knowledge


class DynamicFilter(StaticFilter):
    def __init__(self, sigma, N=None, max_deep=50, counter_update=5):
        self.sigma = float(sigma) 
        self.N0 = int(N) if N else int(5*sigma) #Approximation for one dimension
        self.max_deep = max_deep
        self.N0 = min(self.N0, self.max_deep)
        self.counter_max = int(counter_update) if int(counter_update) < self.N0 else self.N0
        self.clear_data()
        self.calculate_g_vec()

    def clear_data(self):
        self.in_vec = deque(maxlen=self.N0)
        self.counter_update = 0

    def restart(self):
        self.clear_data()
        self.calculate_g_vec()

    def put(self, new_in_value):
        self.in_vec.appendleft(new_in_value)
        self.counter_update += 1
        if self.counter_max < self.counter_update:
            self.counter_update = 0
            self.update_sigma()

    def update_sigma(self):
        data_std = np.std(self.in_vec)
        self.sigma =  self.sigma + data_std if data_std > self.sigma else 0.5*(self.sigma + data_std) #Inspired in Fuzzy intead of Bayes an, because Bayesian takes the least std, but I want sigma=std to grow if data_std gets bigger 
        self.update_N0()
        self.calculate_g_vec()

    def update_N0(self):
        N = min(int(5*self.sigma + .5), self.max_deep)
        self.N0 = N if N > 3 else 3
        temp = deque(maxlen=self.N0)
        limit = min(self.N0, len(self.in_vec))
        for i in range(limit):
            temp.append(self.in_vec[i])
        self.in_vec = temp


