# %% [markdown]
# # Audio classification using convolutional neural networks
# 
# You are the leader of a group of climate scientists who are concerned about the planet's dwindling rain forests. The world loses up to 10 million acres of old-growth rain forests each year, much of it due to illegal logging. Deforestation from this and other causes accounts for about 10% of global carbon emissions. Your team plans to convert thousands of discarded smart phones into solar-powered listening devices and position them throughout the Amazon to transmit alerts in response to the sounds of chainsaws and truck engines. You need software to install on these phones that uses artificial intelligence (AI) to identify such sounds in real time. And you need it fast, because climate change won't wait.
# 
# Audio classification can be performed by converting audio streams into [spectrograms](https://en.wikipedia.org/wiki/Spectrogram), which provide visual representations of spectrums of frequencies as they vary over time, and classifying the spectrograms using [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNNs). The spectrograms below were generated from WAV files with chainsaw sounds in the foreground and rainforest sounds in the background. Let's use Keras to build a CNN that can identify the tell-tale sounds of logging operations and distinguish them from ambient sounds such as wildlife and thunderstorms.
# 
# ![](Images/spectrograms.png)
# 
# This notebook was inspired by the [Rainforest Connection](https://rfcx.org/), which uses recycled Android phones and a TensorFlow model to monitor rain forests for sounds indicative of illegal activity. For more information, see [The fight against illegal deforestation with TensorFlow](https://blog.google/technology/ai/fight-against-illegal-deforestation-tensorflow/) in the Google AI blog. It is just one example of how AI is making the world a better place.

# %% [markdown]
# ## Generate spectrograms
# 
# The "Sounds" directory contains subdirectories named "background," "chainsaw," "engine," and "storm." Each subdirectory contains 100 WAV files. The WAV files in the "background" directory contain rainforest background noises only, while the files in the other subdirectories include the sounds of chainsaws, engines, and thunderstorms overlaid on the background noises. These WAV files were generated by using a soundscape-synthesis package named [Scaper](https://pypi.org/project/scaper/) to combine sounds in the public [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset with rainforest sounds obtained from YouTube.
# 
# The first step is to load the WAV files, use a Python package named [Librosa](https://librosa.org/) to generate spectrogram images from them, load the spectrograms into memory, and prepare them for use in training a CNN. To aid in this process, we'll define a pair of helper functions for creating spectrograms from WAV files and converting all the WAV files in a specified directory into spectrograms.

# %%
import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
# %%matplotlib inline

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)

# %% [markdown]
# Create PNG files containing spectrograms from all the WAV files in the "Sounds/background" directory.

# %%
create_pngs_from_wavs('Sounds/background', 'Spectrograms/background')

# %% [markdown]
# Create PNG files containing spectrograms from all the WAV files in the "Sounds/chainsaw" directory.

# %%
create_pngs_from_wavs('Sounds/chainsaw', 'Spectrograms/chainsaw')

# %% [markdown]
# Create PNG files containing spectrograms from all the WAV files in the "Sounds/engine" directory.

# %%
create_pngs_from_wavs('Sounds/engine', 'Spectrograms/engine')

# %% [markdown]
# Create PNG files containing spectrograms from all the WAV files in the "Sounds/storm" directory.

# %%
create_pngs_from_wavs('Sounds/storm', 'Spectrograms/storm')

# %% [markdown]
# Define two new helper functions for loading and displaying spectrograms and declare two Python lists — one to store spectrogram images, and another to store class labels.

# %%
from keras.preprocessing import image

def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))
        
    return images, labels

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)
        
x = []
y = []

# %% [markdown]
# Load the background spectrogram images, add them to the list named `x`, and label them with 0s.

# %%
images, labels = load_images_from_path('Spectrograms/background', 0)
show_images(images)
    
x += images
y += labels

# %% [markdown]
# Load the chainsaw spectrogram images, add them to the list named `x`, and label them with 1s.

# %%
images, labels = load_images_from_path('Spectrograms/chainsaw', 1)
show_images(images)
    
x += images
y += labels

# %% [markdown]
# Load the engine spectrogram images, add them to the list named `x`, and label them with 2s.

# %%
images, labels = load_images_from_path('Spectrograms/engine', 2)
show_images(images)
    
x += images
y += labels

# %% [markdown]
# Load the storm spectrogram images, add them to the list named `x`, and label them with 3s.

# %%
images, labels = load_images_from_path('Spectrograms/storm', 3)
show_images(images)
    
x += images
y += labels

# %% [markdown]
# Split the images and labels into two datasets — one for training, and one for testing. Then divide the pixel values by 255 and one-hot-encode the labels using Keras's [to_categorical](https://keras.io/api/utils/python_utils/#to_categorical-function) function.

# %%
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MyModel, self).__init__()
        
        # Input shape: (B, 3, 224, 224) analogous to (224, 224, 3) in Keras, but channels-first
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        # After four rounds of pooling with kernel_size=2, 
        # input of 224x224 will be reduced to 224/(2^4)=224/16=14 in each spatial dimension.
        # So final feature map size after the last conv/pool block: (128, 14, 14).
        
        # Flattening 128 * 14 * 14 = 128 * 196 = 25088
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # For training with cross-entropy, no softmax is needed here.
        return x

model = MyModel(num_classes=4)
print(model)



# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming x_train_norm, y_train_encoded, x_test_norm, and y_test_encoded are numpy arrays
# Convert one-hot encoded labels to class indices
y_train_indices = np.argmax(y_train_encoded, axis=1)
y_test_indices = np.argmax(y_test_encoded, axis=1)

# Convert the data to PyTorch tensors
x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
y_train_tensor = torch.tensor(y_train_indices, dtype=torch.long)
x_test_tensor = torch.tensor(x_test_norm, dtype=torch.float32).permute(0, 3, 1, 2)
y_test_tensor = torch.tensor(y_test_indices, dtype=torch.long)

# Create DataLoaders for training and validation
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Define the model, criterion, and optimizer
model = MyModel()  # Ensure MyModel is defined as before
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        correct_train += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = correct_train / len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            correct_val += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(test_loader.dataset)
    val_accuracy = correct_val / len(test_loader.dataset)

    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from fft_conv_pytorch import FFTConv2d  # Importing the FFT-based convolution class

# %%
class MyModelFFT(nn.Module):
    def __init__(self, num_classes=4):
        super(MyModelFFT, self).__init__()  # Correct: use MyModelFFT here
        
        # Using FFTConv2d instead of nn.Conv2d
        self.conv1 = FFTConv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = FFTConv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = FFTConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = FFTConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        # After four rounds of 2x2 pooling:
        # Input: (B, 3, 224, 224)
        # After 4x pooling (2^4 = 16): feature map size = 14x14 if input is 224x224.
        # Channels after last conv block = 128
        # Flatten size = 128 * 14 * 14 = 25088
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass using fft-based convolutions
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits are returned
        return x

# Instantiate the model
model = MyModelFFT(num_classes=4)
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming x_train_norm, y_train_encoded, x_test_norm, and y_test_encoded are numpy arrays
# Convert one-hot encoded labels to class indices
y_train_indices = np.argmax(y_train_encoded, axis=1)
y_test_indices = np.argmax(y_test_encoded, axis=1)

# Convert the data to PyTorch tensors
x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
y_train_tensor = torch.tensor(y_train_indices, dtype=torch.long)
x_test_tensor = torch.tensor(x_test_norm, dtype=torch.float32).permute(0, 3, 1, 2)
y_test_tensor = torch.tensor(y_test_indices, dtype=torch.long)

# Create DataLoaders for training and validation
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Define the model, criterion, and optimizer
model = MyModelFFT(num_classes=4)  # Ensure MyModel is defined as before
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        correct_train += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = correct_train / len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            correct_val += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(test_loader.dataset)
    val_accuracy = correct_val / len(test_loader.dataset)

    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")



