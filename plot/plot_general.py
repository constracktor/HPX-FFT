import os
import numpy as np
from scipy.stats.distributions import chi2

# Greyscale
# black - dark grey - grey - light grey - very light grey
greyscale = ['#000000', '#333333', '#666666', '#999999', '#cccccc']

# CVD accessible colors
#           0 black   1 dark red  2 indigo   3 yellow   4 teal    5 light gray
colors_6 = ['#000000','#c1272d', '#0000a7', '#eecc16', '#008176', '#b3b3b3']
#https://lospec.com/palette-list/krzywinski-colorblind-16
#          0 black    1 d-green  2 m-green 3 l-green  4 d-blue  5 m-blue   6 l-blue  7 d-purple  8 l-purple  9 d-pink  10 l-pink  11 r-brown 12 d-brown 13 l-brown  14 y-brown  15 white
colors_16 = ['#000000','#004949', '#009292', '#24ff24', '#006edb', '#6db6ff', '#b6dbff', '#490092', '#b66dff', '#ff6db6', '#ffb6db', '#920000', '#924900', '#db6d00', '#ffff6d', '#ffffff']

# General printing object
class plot_object:
    def __init__(self, filepath, number_loop):
        self.loop = number_loop
        self.mean = self.compute_operation_from_file(filepath, np.mean)
        self.var = self.compute_operation_from_file(filepath, np.var)
        self.std = self.compute_operation_from_file(filepath, np.std)
        self.median = self.compute_operation_from_file(filepath, np.median)
        self.min = self.compute_operation_from_file(filepath, np.min)
        self.max = self.compute_operation_from_file(filepath, np.max)

    def compute_operation_from_file(self, filepath, operation):
        matrix =  np.genfromtxt(os.path.abspath(filepath), dtype='float', delimiter=';' , skip_header=1)
        n_entries = int(matrix.shape[0] /self.loop)
        matrix_operation = np.zeros((n_entries, matrix.shape[1]))
        for i in range (n_entries):
            matrix_operation[i,:] = operation(matrix[i *self.loop:(i + 1) *self.loop,:], axis=0)
        return matrix_operation
    
    def min_max_error(self, column):
        return [self.median[:, column] - self.min[:, column], self.max[:, column] - self.median[:, column]]

    def confidence_error(self, column, confidence=0.95):
        alpha = 1.0 - confidence
        lower_limit = self.std[:, column] * np.sqrt((self.loop - 1) / chi2.ppf(1 - (alpha / 2), df=(self.loop - 1)))
        upper_limit = self.std[:, column] * np.sqrt((self.loop - 1) / chi2.ppf(alpha / 2, df=(self.loop - 1)))
        return [lower_limit, upper_limit]