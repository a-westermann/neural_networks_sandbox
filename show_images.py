import matplotlib.pyplot as plt


def show_images(images: [], titles: [str]):
    cols = 5
    rows = int(len(images)/cols)
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, titles):
        image = x[0]
        title = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        if title != '':
            plt.title(title, fontsize=15)
        index += 1
    plt.show()

