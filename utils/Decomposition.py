import cv2
import numpy as np
from torchvision import transforms


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # return cv2.LUT(img, gamma_table)
    return cv2.LUT(img.astype(np.uint8), gamma_table)

def adjust_gammma(img_gray):
    # mean = np.mean(img_gray)
    # gamma_val = math.log10(0.5) / math.log10(mean / 255)
    # print(gamma_val)
    image_gamma_correct = gamma_trans(img_gray, 2.0)
    return image_gamma_correct

def init_decomposition(image_cv, image_pil):
    size = (image_cv[0], image_cv[1])
    # image = cv2.resize(image_cv, (512, 512), interpolation=cv2.INTER_CUBIC)
    temp = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    v = temp[:, :, 2]
    cv2.imwrite('./output/output/H.png', v)

    illumination = np.clip(np.asarray([v for _ in range(3)]), 0.000002, 255)
    illumination = illumination.transpose(1, 2, 0)
    k = illumination.astype(np.uint8)
    cv2.imwrite('output/output/illumination.png', k)

    reflection = image_cv
    cv2.imwrite('output/output/reflection.png', reflection)

    return illumination, reflection