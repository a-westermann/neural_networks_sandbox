from mnist_dataloader import MnistDataloader
from show_images import show_images
from random import randrange


dataloader = MnistDataloader('dataset/t10k-labels.idx1-ubyte', 'dataset/t10k-images.idx3-ubyte')
images, labels = dataloader.images, dataloader.labels

rand_nums = [randrange(0, 9999) for _ in range(15)]
rand_images = [images[x] for x in rand_nums]
rand_labels = [f'digit: {labels[x]}' for x in rand_nums]
show_images(rand_images, rand_labels)
