from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from scipy.ndimage.interpolation import rotate
from tsne import tsne
import pylab as Plot
import idx2numpy


# generate new examples with noise
def generate_with_noise(image):
    # rotate the image
    if np.random.rand() < 0.3:
        image = rotate(image, angle=np.random.randint(-15, 15, 1), reshape=False)
    # random noise
    if np.random.rand() < 0.3:
        image[np.random.randint(28, size=1), :] = 0.0
    if np.random.rand() < 0.3:
        image[:, np.random.randint(28, size=1)] = 0.0
    # gets 2 numbers between -3 and 3 (by random)
    [horizontal_jitter, vertical_jitter] = np.random.randint(-3, 3, size=2)
    # horizontal jitter
    if np.random.rand() < 0.3:
        image = np.roll(image, horizontal_jitter, axis=0)
    # vertical jitter
    if np.random.rand() < 0.3:
        image = np.roll(image, vertical_jitter, axis=1)
    return image


input_img = Input(shape=(784,))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(30, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='adadelta', loss='mse')

# read the data
x_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
x_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# create valid and normalize
x_train = x_train.astype('float32') / 255.
x_valid = x_train[-10000:]
x_train = x_train[:50000]
x_test = x_test.astype('float32') / 255.

# reshape the input
x_train_orig = np.copy(x_train)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# calculate the accuracy using nn method
def check_distance(encoded_imgs):
    encoded_len = len(encoded_imgs)
    num_correct = 0.0
    for i in range(encoded_len):
        min = np.inf
        nearest = None
        ed = None
        for j in range(encoded_len):
            if i == j:
                continue
            ed = np.linalg.norm(encoded_imgs[i]-encoded_imgs[j])
            if ed < min:
                min = ed
                nearest = j
        if y_test[i] == y_test[nearest]:
            num_correct += 1
    return num_correct / len(y_test)


def visualize(file_name, encoded_imgs):
    X = encoded_imgs
    labels = y_test
    Y = tsne(X, 2, 50, 20.0)
    Plot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    Plot.show()
    Plot.savefig(file_name, bbox_inches='tight')


def train():
    # Training
    for i in range(10):
        # Copies to not effect the originals
        x_train_temp = np.copy(x_train_orig)

        # Adds noise (jitter)

        for j in range(50000):
            x_train_temp[j] = generate_with_noise(x_train_temp[j])

        x_train_temp = x_train_temp.reshape((len(x_train_temp), np.prod(x_train_temp.shape[1:])))

        autoencoder.fit(x_train_temp, x_train,
                        epochs=1000,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_valid, x_valid))

        encoded_imgs = encoder.predict(x_test)

        result = check_distance(encoded_imgs)
        file_name = 'plt.png'
        visualize(file_name, encoded_imgs)
        print "accuracy: " + str(result)


train()


