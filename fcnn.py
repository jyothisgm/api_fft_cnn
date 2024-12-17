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

# !wget https://surfdrive.surf.nl/files/index.php/s/62Lnr1AtGe9b0v9/download -O face_dataset_64x64.npy

def load_real_samples(scale=False):
    # We load 20,000 samples only to avoid memory issues, you can  change this value
    X = np.load('images_64x64.npy',  fix_imports=True,encoding='latin1')[:20000, :, :, :]
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
grid_plot(dataset[np.random.randint(0, 1000, 9)], name='Faces dataset (64x64x3)', n=3)

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
for epoch in range(10):
    print('\nEpoch: ', epoch)

    # Note that (X=y) when training autoencoders!
    # In this case we only care about qualitative performance, we don't split into train/test sets
    cae.fit(x=dataset, y=dataset, epochs=1, batch_size=64)
    idx = 18653
    samples = dataset[idx:idx+9]
    reconstructed = cae.predict(samples)
    grid_plot(samples, epoch, name='Original', n=3, save=False)
    grid_plot(reconstructed, epoch, name='Reconstructed', n=3, save=False)


#%% FFT EXPERIMENT
input_data = dataset[0:1].astype(np.float32)
encoder = cae.layers[0]
layers = encoder.layers

ishape0 = layers[0].batch_shape
ishapes= [ishape0]+[l.output.shape for l in layers]

c1 = layers[1]
output = c1.call(input_data)
w, b = c1.weights

#%%
from numpy.fft import fft2,ifft2,fft,ifft

def fft_convolve2d(channel, kernel):
    H, W, = channel.shape  # (64, 64)
    kh, kw = kernel.shape  # (3, 3)
    convolved = np.zeros_like(channel)

    kernel_padded = np.zeros_like(channel)
    kernel_padded[:kh, :kw] = kernel
    
    # FFT of the image and the padded kernel
    fft_img = fft2(channel)
    fft_kernel = fft2(kernel_padded)
    fft_result = fft_img * fft_kernel
    
    # Inverse FFT to get the spatial domain result
    convolved = ifft2(fft_result).real
    return convolved

#%%
from scipy.signal import convolve2d  # For easy comparison

def spatial_convolve2d(image, kernel):
    """ og conv2d for comparison"""
    convolved = np.zeros_like(image)
    
    # Flip the kernel for convolution (not required for correlation)
    # flipped_kernel = np.flip(kernel)
    
    img_channel = image[:, :]
    convolved = convolve2d(img_channel, kernel, mode='same', boundary='fill', fillvalue=0)
    
    return convolved

#%%
# Example usage
image = dataset[2]
c = 0
l0_weights = w[:,:,:,0]
kernel = l0_weights[:,:,c]
fmap = image[:,:,c]
# Compute the spatial convolution
spatial_result = spatial_convolve2d(fmap,kernel)

# Compute the FFT convolution
fft_result = fft_convolve2d(fmap, kernel)

# Check if results are approximately the same
print("Difference between FFT and spatial results:", np.max(np.abs(spatial_result - fft_result)))

plt.figure(figsize=(15, 5))

# Original image (sum over channels for visualization)
plt.subplot(1, 4, 1)
plt.imshow(image )
plt.title("Original Image")
plt.axis('off')

# FFT-based convolution result
f = (fft_result-fft_result.min())/(fft_result.max()-fft_result.min())
plt.subplot(1, 4, 2)
plt.imshow(f,cmap='gray')
plt.title("FFT-based Convolution")
plt.axis('off')

# Spatial convolution result
s = (spatial_result-spatial_result.min())/(spatial_result.max()-spatial_result.min())
plt.subplot(1, 4, 3)
plt.imshow(s,cmap='gray')#,vmin = spatial_result.min(),vmax=spatial_result.max())
plt.title("Spatial Convolution")
plt.axis('off')

# # Diff of methods
plt.subplot(1, 4, 4)
diff = spatial_result-fft_result
d = (diff-diff.min())/(diff.max()-diff.min())
plt.imshow(d,cmap='gray')
plt.title("Diff")
plt.axis('off')

plt.tight_layout()
plt.show()
# %%
def fft_channels(weights,fmap):
    H,W,C = fmap.shape
    h,w,c = weights.shape
    convs = list()
    for i in range(weights.shape[-1]):
        channel = fmap[:,:,i]
        kernel = weights[:,:,i]
        conv = fft_convolve2d(channel,kernel)
    convs.append(conv)
    return convs
res = fft_channels(l0_weights,image)

# %%
