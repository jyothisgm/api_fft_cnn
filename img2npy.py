#%%
import os
import cv2
import numpy as np
from tqdm import tqdm

# Function to load all .jpg images from a directory and subdirectories
def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in tqdm(files):
            if file.lower().endswith((".jpg",".png")):
                file_path = os.path.join(root, file)
                try:
                    # Read image
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Reads the image in color
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img,(64,64))
                    if img is not None:
                        images.append(img)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
    return images

# Main folder path
main_folder = "images"  # Replace with your folder path
cat_folder = "cat_images"
folder = cat_folder
print(f"Loading images from {folder} folder")
images_resized = load_images_from_folder(folder)
#%%
# Convert the list of images to a numpy array
images_array = np.array(images_resized, dtype=np.uint8)  # Use dtype=object for images of varying dimensions
print(images_array.shape)
#%%
# Save the numpy array to a .npy file
#%%
output_file = "cats_images_64x64.npy"
np.save(output_file,images_array)

print(f"Saved {len(images_resized)} images to {output_file}")


# %%
