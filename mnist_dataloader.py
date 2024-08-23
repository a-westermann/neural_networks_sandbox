import struct
from array import array
import numpy as np
from numpy import reshape

class MnistDataloader:
    def __init__(self, labels_path: str, images_path: str):
        self.images, self.labels = self.read_images_labels(labels_path, images_path)

    @staticmethod
    def read_images_labels(labels_path: str, images_path: str) -> ([], []):
        labels = []
        with open(labels_path, 'rb') as labels_file:
            magic, size = struct.unpack('>II', labels_file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, received {magic}')
            labels = array('B', labels_file.read())

        with open(images_path, 'rb') as images_file:
            magic, size, rows, cols = struct.unpack('>IIII', images_file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, received {magic}')
            image_data = array('B', images_file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            image = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            image = np.reshape(image, (28, 28))
            images[i][:] = image

        return images, labels
