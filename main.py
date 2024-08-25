import random


from mnist_dataloader import MnistDataloader
from show_images import show_images
from random import randrange
from neural_network import NeuralNetwork


def show_random_images():
    rand_nums = [randrange(0, 9999) for _ in range(15)]
    rand_images = [images[x] for x in rand_nums]
    rand_labels = [f'digit: {labels[x]}' for x in rand_nums]
    show_images(rand_images, rand_labels)


def write_weihts_bias(weight_arr, bias_arr):
    with open('weights', 'w') as w_file:
        w_str = weight_arr.__repr__().replace('array(', '').replace(')', '')
        w_file.write(w_str)
    with open('bias', 'w') as b_file:
        b_str = bias_arr.__repr__().replace('array(', '').replace(')', '')
        b_file.write(b_str)


shape = (28, 28)
dataloader = MnistDataloader('dataset/t10k-labels.idx1-ubyte', 'dataset/t10k-images.idx3-ubyte',
                             shape)
images, labels = dataloader.images, dataloader.labels

nn = NeuralNetwork(0.1, shape)
# nn.load_weights_bias('weights', 'bias')
for i in range(len(images)):
    bias, weights = nn.compute_gradients(images[i], labels[i])
    nn.update_parameters(bias, weights)
    if i % 100 == 0:
        print(f'Predicting for : {labels[i]}')
        print(nn.predict(images[i])[0][0])

weights, bias = nn.weights, nn.bias
write_weihts_bias(weights, bias)
