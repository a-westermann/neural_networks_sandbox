

class MnistDataloader:
    def __init__(self):
        self.dataset = self.read_images_labels('mnist_digits.idx3-ubyte')


    def read_images_labels(self, path: str) -> ([], []):
        labels = []
        with (open)
