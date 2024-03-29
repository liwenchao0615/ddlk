import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

noise_type = 'gauss' # Either 'gauss' or 'poiss'
noise_level = 50  # Pixel range is 0-255 for Gaussian, and 0-1 for Poission
def add_noise(x,noise_level):

    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape)
        noisy = torch.clamp(noisy,0,1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x)/noise_level

    return noisy

def init_input(original_illumination, original_reflection):
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    illumination_tensor = transform(original_illumination).unsqueeze(0)
    reflection_tensor = transform(original_reflection).unsqueeze(0)
    illumination_tensor_addnoise = add_noise(illumination_tensor, noise_level)
    reflection_tensor_addnoise = add_noise(reflection_tensor, noise_level)
    #输出加入噪声的图像
    illumination_tensor_addnoise_output = illumination_tensor_addnoise.squeeze(0).numpy()
    illumination_tensor_addnoise_output = illumination_tensor_addnoise_output.transpose(1, 2, 0)
    illumination_addnoise = np.clip(illumination_tensor_addnoise_output*255, 0, 255).astype(np.uint8)
    cv2.imwrite('./output/output/illumination_addnoise.png', illumination_addnoise)

    reflection_tensor_addnoise_output = reflection_tensor_addnoise.squeeze(0).numpy()
    reflection_tensor_addnoise_output = reflection_tensor_addnoise_output.transpose(1, 2, 0)
    reflection_addnoise = np.clip(reflection_tensor_addnoise_output*255, 0, 255).astype(np.uint8)
    cv2.imwrite('./output/output/reflection_addnoise.png', reflection_addnoise)

    return illumination_tensor_addnoise, reflection_tensor_addnoise