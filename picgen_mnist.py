from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model, load_model, model_from_json, Sequential
from keras import backend as K
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt


#mean=np.load("mean.npy")
#mean=np.load("red_mean.npy")
mean=np.load("mean_network_1m.npy")
#cov=np.load("cov.npy")
#cov=np.load("red_cov.npy")
cov=np.load("cov_network_1m.npy")
#print(mean.shape)
#print(cov.shape)
C=mean.shape[0]
D=cov.shape[1]

latent_dim=D


# load json and create model
#json_file = open("autoencoder_mnist.json", "r")
json_file = open("autoencoder_network_1m.json", "r")
autoencoder_json = json_file.read()
json_file.close()
autoencoder = model_from_json(autoencoder_json)

# load weights into new model
#autoencoder.load_weights("autoencoder_mnist.h5")
autoencoder.load_weights("autoencoder_network_1m.h5")
print("Loaded model from disk")

#autoencoder.summary()


encoder = Model(autoencoder.input, autoencoder.layers[-10].output)

#encoder.summary()


decoder_input = Input(shape=(latent_dim, ))

x = autoencoder.layers[-9](decoder_input)
x = autoencoder.layers[-8](x)
x = autoencoder.layers[-7](x)
x = autoencoder.layers[-6](x)
x = autoencoder.layers[-5](x)
x = autoencoder.layers[-4](x)
x = autoencoder.layers[-3](x)
x = autoencoder.layers[-2](x)
x = autoencoder.layers[-1](x)
decoder = Model(decoder_input, x)

#decoder.summary()


n = 8  #how many predictions of each component we will display

#generate randoms
randoms=np.zeros([C, n, D])
for k in range(C):
    randoms[k]=multivariate_normal(mean[k].flatten(), cov[k]/10, n)
randoms=randoms.reshape(C*n, D)


prediction = decoder.predict(randoms)
#print(prediction)


plt.figure(figsize=(10, 10))
for i in range(n):
    for k in range(C):
        # display reconstruction
        ax = plt.subplot(C, n, k * n + i + 1)
        plt.imshow(prediction[k * n + i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()