import numpy as np
from PIL import Image
import os


class BinaryImageConverter:
    def __init__(self, tensor):
        self.tensor = tensor

    def convert_and_save_images(self, save_path):
        # Reshape the tensor to (batch_size, width, height)
        batch_size, _, width, height = self.tensor.shape
        images = self.tensor.reshape(batch_size, width, height)

        for i in range(batch_size):

            image = images[i]
            image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='L')

            image_path = os.path.join(save_path, f'image_{i}.png')

            # Save the image as PNG
            image.save(image_path)