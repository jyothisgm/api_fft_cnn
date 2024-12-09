#%%
import tensorflow as tf
import math

class SpectralConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)

        # Separate real and imaginary parts
        self.weights1_real = self.add_weight(
            shape=(in_channels, out_channels, modes1, modes2),
            initializer=tf.keras.initializers.RandomUniform(-self.scale, self.scale),
            dtype=tf.float32,
            trainable=True,
        )
        self.weights1_imag = self.add_weight(
            shape=(in_channels, out_channels, modes1, modes2),
            initializer=tf.keras.initializers.RandomUniform(-self.scale, self.scale),
            dtype=tf.float32,
            trainable=True,
        )
        self.weights2_real = self.add_weight(
            shape=(in_channels, out_channels, modes1, modes2),
            initializer=tf.keras.initializers.RandomUniform(-self.scale, self.scale),
            dtype=tf.float32,
            trainable=True,
        )
        self.weights2_imag = self.add_weight(
            shape=(in_channels, out_channels, modes1, modes2),
            initializer=tf.keras.initializers.RandomUniform(-self.scale, self.scale),
            dtype=tf.float32,
            trainable=True,
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights_real, weights_imag):
        # Combine real and imaginary parts into a complex tensor
        weights = tf.complex(weights_real, weights_imag)
        return tf.einsum("bixy,ioxy->boxy", input, weights)

    def call(self, x):
        batchsize = tf.shape(x)[0]

        # Compute Fourier coefficients
        x_ft = tf.signal.fft2d(tf.cast(x, dtype=tf.complex64))

        # Prepare output tensor for the relevant Fourier modes
        out_ft = tf.zeros(
            (batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1),
            dtype=tf.complex64,
        )

        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1_real,
            self.weights1_imag,
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2_real,
            self.weights2_imag,
        )

        # Return to physical space
        x = tf.signal.ifft2d(out_ft)
        x = tf.math.real(x)  # Convert back to real values
        return x


class SpectralConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2DTranspose, self).__init__()

        """
        2D Transpose Fourier layer. Performs FFT, linear transform, and Inverse FFT,
        designed for upsampling in the Fourier domain.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.add_weight(
            shape=(in_channels, out_channels, modes1, modes2),
            initializer=tf.keras.initializers.RandomUniform(-self.scale, self.scale),
            dtype=tf.complex64,
            trainable=True,
        )
        self.weights2 = self.add_weight(
            shape=(in_channels, out_channels, modes1, modes2),
            initializer=tf.keras.initializers.RandomUniform(-self.scale, self.scale),
            dtype=tf.complex64,
            trainable=True,
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return tf.einsum("bixy,ioxy->boxy", input, weights)

    def call(self, x):
        batchsize = tf.shape(x)[0]

        # Compute Fourier coefficients
        x_ft = tf.signal.fft2d(tf.cast(x, dtype=tf.complex64))

        # Prepare output tensor for the relevant Fourier modes
        out_ft = tf.zeros(
            (batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1),
            dtype=tf.complex64,
        )

        # Perform the transpose operation by reversing the Fourier mode multiplications
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], tf.math.conj(self.weights1)
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], tf.math.conj(self.weights2)
        )

        # Return to physical space
        x = tf.signal.ifft2d(out_ft)
        x = tf.math.real(x)  # Convert back to real values
        return x

#%%
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%%
def load_real_samples(scale=False):
    # We load 20,000 samples only to avoid memory issues, you can  change this value
    X = np.load('face_dataset_64x64.npy',  fix_imports=True,encoding='latin1')[:20000, :, :, :]
    # Scale samples in range [-127, 127]
    if scale:
        X = (X - 127.5) * 2
    return X / 255.

# We will use this function to display the output of our models throughout this notebook
def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        filename = 'results/generated_plot_e%03d_f.png' % (epoch+1)
        plt.savefig(filename)
        plt.close()
    plt.show()


dataset = load_real_samples()
grid_plot(dataset[np.random.randint(0, 1000, 9)], name='Fliqr dataset (64x64x3)', n=3)

#%%

from tensorflow.keras.layers import Dense, Flatten, Reshape

def build_conv_net(in_shape, out_shape, n_downsampling_layers=4, filters=128, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    input = tf.keras.Input(shape=in_shape)
    x = SpectralConv2d(in_channels=3, out_channels=128, modes1=math.floor(image_size[0]/2) + 1, modes2=math.floor(image_size[1]/2) + 1)(input)

    for _ in range(n_downsampling_layers):
        x = SpectralConv2d(in_channels=3, in_channels=128, 
                           modes1=in_shape[0]/2 + 1, modes1=in_shape[1]/2 + 1)(x)

    x = Flatten()(x)
    x = Dense(out_shape, activation=out_activation, name='enc_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Encoder')

    model.summary()
    return model


def build_deconv_net(latent_dim, n_upsampling_layers=4, filters=128, activation_out='sigmoid'):
    """
    Build a deconvolutional network for decoding/upscaling latent vectors

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that
    were used in the downsampling network and the Conv2DTranspose layers are used instead of Conv2D layers.
    Using identical layers and hyperparameters ensures that the dimensionality of our output matches the
    shape of our input images.
    """
    input = tf.keras.Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 64, input_dim=latent_dim, name='dec_input')(input)
    x = Reshape((4, 4, 64))(x) # This matches the output size of the downsampling architecture

    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    for i in range(n_upsampling_layers):
        x = SpectralConv2DTranspose(filters=filters, **default_args)(x)

    # This last convolutional layer converts back to 3 channel RGB image
    x = SpectralConv2d(filters=3, kernel_size=(3,3), padding='same', activation=activation_out, name='dec_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Decoder')
    model.summary()
    return model

#%%
def build_convolutional_autoencoder(data_shape, latent_dim, filters=128):
    encoder = build_conv_net(in_shape=data_shape, out_shape=latent_dim, filters=filters)
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid', filters=filters)

    # We connect encoder and decoder into a single model
    autoencoder = tf.keras.Sequential([encoder, decoder])

    # Binary crossentropy loss - pairwise comparison between input and output pixels
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

    return autoencoder


# Defining the model dimensions and building it
image_size = dataset.shape[1:]
latent_dim = 512
num_filters = 128
cae = build_convolutional_autoencoder(image_size, latent_dim, num_filters)

#%%
for epoch in range(5):
    print('\nEpoch: ', epoch)

    # Note that (X=y) when training autoencoders!
    # In this case we only care about qualitative performance, we don't split into train/test sets
    cae.fit(x=dataset, y=dataset, epochs=1, batch_size=64)

    samples = dataset[:9]
    reconstructed = cae.predict(samples)
    grid_plot(samples, epoch, name='Original', n=3, save=False)
    grid_plot(reconstructed, epoch, name='Reconstructed', n=3, save=False)
#%%
