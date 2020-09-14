"""
Sample Gaussian Filter for a casual system. Based on equation (1a) from article.

"""
import queue


class CasualFilter:
    def __init__(self, sigma, N=None, counter_update=5):
        self.sigma = float(sigma) if sigma > 0.0 else 2.3e-16 
        self.N0 = int(N) if N else int(5*sigma) #Approximation for one dimension
        self.max_counter = int(counter_update) if int(counter_update) > 1 else 5
        self.restart()

    def clear_data(self):
        self.in_vec = queue.Queue(maxlen=self.N0)
        self.counter_update = 0

    def calculate_g_vec():
        temp = 1 / ( sqrt(2*math.pi)*self.sigma)
        self.g_vec = [temp]
        for i in range(1, self.N0):
            self.g_vec.append(temp * math.exp(-0.5*i/(self.sigma**2)))

    def restart(self):
        self.clear_data()
        self.calculate_g_vec()

    def out_value(self, new_in_value):
        self.put(new_in_value)
        return self.conv_data()

    def put(self, new_in_value):
        self.in_vec.append(new_in_value)
        self.counter_update += 1
        if self.counter_max < self.counter_update:
            self.counter_update = 0
            self.update_sigma()

    def conv_data(self):
        sum_value = 0
        for in_k, g_k in zip(self.in_vec, self.g_vec):
            sum_value += in_k * g_k
        return sum_value

    def update_sigma(self):
        #Baysian Recursive formula
        data_std = self.input_std()
        self.sigma = 1 / ( self.sigma**-1 + data_std**-1)
        self.calculate_g_vec()

    def input_std()
        raise NotImplemented
