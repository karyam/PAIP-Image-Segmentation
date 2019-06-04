import numpy as np


class PoolingLayer:
    """
        Class which represents a basic pooling layer

        Attributes:
            in_depth: input_shape[0] - no of color channels for the input image
            in_height: input_shape[1] - height of the input image
            in_width: input_shape[2] - width of the input image

            pool_size: the size of pooling applied
            stride: the amount of striding

            out_length: the length of the reshaped output vector
            out_height: the height of the output image
            out_width: the width of the output image
            out: the output data after pooling is applied

            arg_max: matrix of 2d coordinates of the indexes of max activation

        """

    def __init__(self, input_shape, pool_size, stride=2):
        # input data
        self.in_depth = input_shape[0]
        self.in_height = input_shape[1]
        self.in_width = input_shape[2]

        # hyper-parameters
        self.pool_size = pool_size
        self.stride = stride

        # output data
        self.out_height = int((self.in_height - self.pool_size) / self.stride + 1)
        self.out_width = int((self.in_width - self.pool_size) / self.stride + 1)
        self.out_length = self.out_width*self.out_height
        self.out = np.zeros((self.in_depth, self.out_height, self.out_width))

        self.arg_max = np.empty(self.in_depth, self.out_height, self.out_width, 2)

    def pooling(self, input_data):

        """
        Function which implements the max pooling operation.

        Worth noting that there are only two commonly seen
        variations of the max pooling layer found in practice:
        A pooling layer with F=3,S=2 (also called overlapping pooling),
        and more commonly F=2,S=2. Pooling sizes with larger receptive fields are too destructive

        However, there is a tendency to discard pooling layers in future architectures
        e.g GANs or VAEs.

        Args:
            input_data: input data received by the pooling layer

        """

        # reshape output and max indexes with flattened length
        self.out = self.out.reshape((self.in_depth, self.out_length))
        self.arg_max = self.arg_max.reshape((self.in_depth, self.out_length))

        for i in range(self.in_depth):
            row, col = 0, 0

            for j in range(self.out_length):
                patch = input_data[i][row:row+self.pool_size[0], col:col+self.pool_size[0]]
                max_value = np.amax(patch)
                self.out[i][j] = max_value
                max_index = np.where(patch == np.max(patch))
                if len(max_index[0]) > 1:
                    max_index = [max_index[0][0], max_index[1][0]]

                # apply patch offset
                max_index = int(max_index[0])+row, int(max_index[0])+col

                self.arg_max[i][j] = max_index

                col += self.stride

                if col >= self.in_width:
                    col = 0
                    row += self.stride

        self.out = self.out.reshape((self.in_depth, self.out_height, self.out_width))
        self.arg_max = self.arg_max.reshape(self.in_depth, self.out_height, self.out_width, 2)


