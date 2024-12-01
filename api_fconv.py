# -*- coding: utf-8 -*-
#%%
# 0 impots
# !pip install 'tensorflow[and-gpu]'==2.14 ##

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%%
# 1 data
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True) ## If you want to use your google drive


def load_real_samples(scale=False):
    # We load 20,000 samples only to avoid memory issues, you can  change this value
    X = np.load('cats_images_64x64.npy',  fix_imports=True,encoding='latin1')[:20000, :, :, :]
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
grid_plot(dataset[np.random.randint(0, 1000, 9)], name='Gato dataset (64x64x3)', n=3)

#%%
# 2 model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape

def build_conv_net(in_shape, out_shape, n_downsampling_layers=4, filters=128, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    input = tf.keras.Input(shape=in_shape)
    x = Conv2D(filters=filters, name='enc_input', **default_args)(input)

    for _ in range(n_downsampling_layers):
        x = Conv2D(**default_args, filters=filters)(x)

    x = Flatten()(x)
    x = Dense(out_shape, activation=out_activation, name='enc_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Encoder')

    model.summary()
    return model


def build_deconv_net(latent_dim, n_upsampling_layers=4, filters=128, activation_out='sigmoid'):
    input = tf.keras.Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 64, input_dim=latent_dim, name='dec_input')(input)
    x = Reshape((4, 4, 64))(x) # This matches the output size of the downsampling architecture

    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    for i in range(n_upsampling_layers):
        x = Conv2DTranspose(filters=filters, **default_args)(x)

    # This last convolutional layer converts back to 3 channel RGB image
    x = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation=activation_out, name='dec_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Decoder')
    model.summary()
    return model

#%%
# CAE example
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


## Training the Convolutional autoencoder to reconstruct images
for epoch in range(5):
    print('\nEpoch: ', epoch)
    cae.fit(x=dataset, y=dataset, epochs=1, batch_size=64)
    idx = 18653
    samples = dataset[idx:idx+9]
    reconstructed = cae.predict(samples)
    grid_plot(samples, epoch, name='Original', n=3, save=False)
    grid_plot(reconstructed, epoch, name='Reconstructed', n=3, save=False)


#%% extract decoder

decoder = cae.layers[1]
input_shape = decoder.input_shape[1:]
decoder.summary()

random_data = 0.8*np.random.random(size=(9,*input_shape))
pred = decoder.predict(random_data)
grid_plot(pred)

#%%
import numpy as np
from scipy.signal import convolve2d

def convolve_fft(channel,kernel):
    kernel_padded = np.zeros_like(channel)
    kernel_padded[:kernel.shape[0],:kernel.shape[1]] = kernel
    #fft
    fft_kernel = np.fft.fft2(kernel_padded)
    fft_channel = np.fft.fft2(channel)
    fft_conv = fft_kernel*fft_channel
    #ifft
    conv = np.fft.ifft2(fft_conv)
    conv_real = conv.real
    return conv_real

#%%
model = cae.layers[0]
layers = model.layers

shape_inputs = [l.output.shape for l in layers[:-1]]
shape_outputs = [l.output.shape for l in layers[1:]]

conv_layers = layers[1:6]

weights = [l.get_weights()[0] for l in conv_layers]
biases = [l.get_weights()[1] for l in conv_layers]

#%%
import matplotlib.pyplot as plt

def plot_diff(kernel,channel):
    input_tensor = image
    fft_image = convolve_fft(channel,kernel)
    spatial_image = convolve2d(channel,kernel,mode='same')
    diff = fft_image-spatial_image
    # Plot the results
    plt.figure(figsize=(15, 5))

    # Original image in RGB
    plt.subplot(1, 4, 1)
    plt.imshow(input_tensor)  # Display as RGB
    plt.title("Original Image ")
    plt.axis('off')

    # FFT-based convolution result
    plt.subplot(1, 4, 2)
    plt.imshow(fft_image, cmap='gray')  # Grayscale for single-channel output
    plt.title("FFT-based Convolution")
    plt.axis('off')

    # Spatial convolution result
    plt.subplot(1, 4, 3)
    plt.imshow(spatial_image, cmap='gray')  # Grayscale for single-channel output
    plt.title("Spatial Convolution")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(diff, cmap='gray')  # Grayscale for single-channel output
    plt.title("Diff")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# choose image and kernel
# from layer 0 get all x and y for channel 0 for neuron 10
kernel = weights[0][:,:,0,10]
bias = biases[0][10]
image = dataset[666]
channel = image[:,:,0]

fft_image = bias + convolve_fft(channel, kernel)
spatial_image = bias + convolve2d(channel, kernel, mode='same')
diff = fft_image-spatial_image
for i in range(3):
    plot_diff(weights[0][:,:,i,10],image[:,:,i])
#%%

def fft_convolve_layer(input_tensor, kernels):
    H, W, C = input_tensor.shape
    k_h, k_w, C_k, K = kernels.shape
    assert C == C_k, "Input channels must match kernel channels."
    output = np.zeros((H, W, K))
    for k in range(K):  # For each kernel
        for c in range(C):  # For each input channel
            kernel = kernels[:, :, c, k]
            channel = input_tensor[:,:,c]
            fft_conv = convolve_fft(kernel, channel)
            conv = convolve2d(channel, kernel, mode='same')
            print("compare!") 
        output[:, :, k] = fft_conv
    
    return output

