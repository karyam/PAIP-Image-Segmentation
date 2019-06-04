import numpy as np
from func_util import *

class ConvolutionalLayer:
    """
    Class which represents a basic convolutional layer

    Attributes:

        in_depth: input_shape[0] - no of color channels for the input image
        in_height: input_shape[1] - height of the input image
        in_width: input_shape[2] - width of the input image

        filter_size: the size of the layer's weights (filers)
        stride: the amount of striding
        no_filters: number of filers
        padding: amount of padding

        weights: the parameters to be learned during training; randomly initialized
        bias: list of 1 values for each filter, representing the bias during training

    """

    def __init__(self, input_shape, filter_size, no_filters, stride, padding=0):
        # input data
        self.in_depth = input_shape[0]
        self.in_height = input_shape[1]
        self.in_width = input_shape[2]

        # hyper-parameters
        self.filer_size = filter_size
        self.stride = stride
        self.no_filters = no_filters
        self.padding = padding

        self.weights = np.random.rand(self.no_filters, input_shape[0], filter_size, filter_size)
        self.bias = np.random.rand(self.no_filters, 1)

        # output data
        self.out_height = int((self.in_height - self.filer_size + 2*self.padding)/self.stride + 1)
        self.out_width = int((self.in_width - self.filer_size + 2*self.padding)/self.stride + 1)
        self.out = np.zeros((self.no_filters, self.out_height, self.out_width))
        self.out_activated = np.zeros((self.no_filters, self.out_height, self.out_width))

    def convolution(self, input_data):

        # flatten rows and columns into one long array so output is no_filters*(out_height*out_width)
        self.out = self.out = self.out.reshape((self.no_filters, self.out_height*self.out_width))
        self.out_activated = self.out_activated = self.out.reshape((self.no_filters, self.out_height * self.out_width))

        out_length = self.out_height*self.out_width

        for i in range(self.no_filters):
            row, col = 0, 0

            # Slide the filter across the image; loop until
            # every value of the flattened output was computed

            for j in range(out_length):
                # Compute the dot product of the current
                # filer and the current patch of the image

                dot_product = input_data[:, row:row+self.filer_size, col:col+self.filer_size] * self.weights[i]
                sum = np.sum(dot_product)
                self.out[i][j] = sum + self.bias[i]
                self.out_activated[i][j] = sigmoid(self.out[i][j])

                col += self.stride

                # check if at the end of the row
                if col + self.filer_size - self.stride >= self.in_width:
                    col = 0
                    row += self.stride
        # reshape output back into the correct shape
        self.out = self.out.reshapeself.out = self.out.reshape((self.no_filters, self.out_height*self.out_width))
        self.out_activated = self.out_activated.reshape((self.no_filters, self.out_height, self.out_width))


