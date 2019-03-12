from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras import losses
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det


latent_dim = 6
#constant in the custom loss function
mu = 0.000001
batch_size = 100
num_epochs = 50

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Flatten()(x)
encoded = Dense(latent_dim, activation='relu')(x)
x = Dense(4*4*8, activation='relu')(encoded)
x = Reshape((4, 4, 8))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


#input_img=(?, 28, 28, 1), encoded=(?, 6), decoded=(?, 28, 28, 1)
autoencoder = Model(inputs=input_img, outputs=[encoded, decoded])
autoencoder.load_weights("autoencoder_mnist.h5")

encoder = Model(input_img, encoded)

#load the tensor containing the mean of the components of the gaussian mixture, shape=(C, D, 1)
mean = np.load("red_mean.npy")
#load the tensor containing the cov matrices of the components, shape=(C, D, D)
cov = np.load("red_cov.npy")
mean = mean.astype('float32')
cov = cov.astype('float32')

#inverses of the covariance matrices of the components ,shape=(1, C, D, D)
invcov = K.expand_dims(K.tf.constant(inv(cov)), axis=0)
#constant function of the determinants of the cov matrices, shape=(1, C, 1, 1)
dets = K.expand_dims(K.expand_dims(K.expand_dims(K.tf.constant(1. / np.sqrt(2 * np.pi * det(cov))), axis=0), axis=-1), axis=-1)
#shape=(1, C, D, 1)
mean_tens = K.expand_dims(K.tf.constant(mean), axis=0)


def mixture_loss(y_true):

    #latent points in the batch, shape=(batch_size, 1, D, 1)
    y_lat = K.expand_dims(K.expand_dims(y_true, axis=-2), axis=-1)
    invcov_bs = K.repeat_elements(invcov, batch_size, axis=0)
    #print(K.dot K.tf.transpose(y_lat - mean_tens, perm=[0, 1, 3, 2]) @ invcov)
    #likelihoods of the batch points, shape=(batch_size, D)
    likel = K.tf.squeeze(dets * K.exp(-0.5 * K.tf.transpose(y_lat - mean_tens, perm=[0, 1, 3, 2]) @ invcov_bs @ (y_lat - mean_tens)), axis=[-2, -1])
    #sum of these likelihoods for the batch
    likel_sum = K.sum(likel, axis=-1)
    return K.sum(likel_sum, axis=-1)


#using a linear combination of the binary crossentropy and the likelihood sum makes sure that the model doesn't collapse but the points are drawn towards the component centers
custom_loss = losses.binary_crossentropy(input_img, decoded) - mu * mixture_loss(encoded)
#custom_loss = losses.binary_crossentropy(input_img, decoded)
autoencoder.add_loss(custom_loss)

autoencoder.compile(optimizer='adadelta')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

#train the model
autoencoder.fit(x_train,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
                #validation_data=(x_test, x_test)
                )

#save the latent points of the trained model
np.save("latent_points_network", encoder.predict(x_train))

# serialize model to JSON
encoder_json = encoder.to_json()
with open("encoder_network.json", "w") as json_file:
    json_file.write(encoder_json)
# serialize weights to HDF5
encoder.save_weights("encoder_network.h5")
print("Saved encoder to disk")

# serialize model to JSON
autoencoder_json = autoencoder.to_json()
with open("autoencoder_network.json", "w") as json_file:
    json_file.write(autoencoder_json)
# serialize weights to HDF5
autoencoder.save_weights("autoencoder_network.h5")
print("Saved autoencoder to disk")