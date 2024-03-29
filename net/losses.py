from colorsys import rgb_to_yiq

import cv2
from torch.cuda import device

from utils.downsampler import pair_downsampler
from .layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
from .downsampler import * 
from torch.nn import functional
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def normalize01(img):
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)

# Gaussian Kernel Initialization
g_kernel_size = 5
g_padding = 2
sigma = 3
kx = cv2.getGaussianKernel(g_kernel_size,sigma)
ky = cv2.getGaussianKernel(g_kernel_size,sigma) #应该得到一个高斯核 5x1，标注差是3

reffac = 1
gaussian_kernel = np.multiply(kx, np.transpose(ky))
gaussian_kernel = torch.FloatTensor(gaussian_kernel).float().unsqueeze(0).unsqueeze(0).to(device)



def illumination_smooth_loss(image, illumination):
    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
    max_rgb, _ = torch.max(image, 1)
    max_rgb = max_rgb.unsqueeze(1)
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    weight_h = 1/(F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w
    max_rgb.detach()
    return loss_h.sum() + loss_w.sum() + torch.norm(illumination-max_rgb, 1)

def reflectance_smooth_loss(image, illumination, reflectance):
    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_reflect_h, gradient_reflect_w = gradient(reflectance)
    weight = 1/(illumination*gradient_gray_h*gradient_gray_w+0.0001)
    weight = normalize01(weight)
    weight.detach()
    loss_h = weight * gradient_reflect_h
    loss_w = weight * gradient_reflect_w
    refrence_reflect = image/illumination
    refrence_reflect.detach()
    return loss_h.sum() + loss_w.sum() + reffac*torch.norm(refrence_reflect - reflectance, 1)

def gradient(img): #img is a Tensor (1,3,640,480)
    height = img.size(2)  #640
    width = img.size(3)   #480
    #一种求梯度的算子
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w

class L_exp(nn.Module):

    def __init__(self, patch_size):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val

    def forward(self, x, mean_val):
        # b,c,h,w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        meanTensor = torch.FloatTensor([mean_val]).cuda()

        d = torch.mean(torch.pow(mean - meanTensor, 2))
        return d

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_down1, src_down2, dst, down1_dst, down2_dst):
        loss_res = 0.5 * ((down1_dst - src_down2) ** 2 + (down2_dst - src_down1) ** 2)
        dst_down1, dst_down2 = pair_downsampler(dst)
        loss_cons = 0.5 * ((down1_dst - dst_down1) ** 2 + (down2_dst - dst_down2) ** 2)
        return (loss_res + loss_cons).mean()

class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


class ExtendedL1Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, b, mask):
        normalizer = self.l1(mask, torch.zeros(mask.shape).cuda())
        # if normalizer < 0.1:
        #     normalizer = 0.1
        c = self.l1(mask * a, mask * b) / normalizer
        return c


class NonBlurryLoss(nn.Module):
    def __init__(self):
        """
        Loss on the distance to 0.5
        """
        super(NonBlurryLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x):
        return 1 - self.mse(x, torch.ones_like(x) * 0.5)


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.gray_scale = GrayscaleLayer()
        self.mse = nn.MSELoss().cuda()

    def forward(self, x, y):
        x_g = self.gray_scale(x)
        y_g = self.gray_scale(y)
        return self.mse(x_g, y_g)


class GrayLoss(nn.Module):
    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)


class GradientLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)


class YIQGNGCLoss(nn.Module):
    def __init__(self, shape=5):
        super(YIQGNGCLoss, self).__init__()
        self.shape = shape
        self.var = VarianceLayer(self.shape, channels=1)
        self.covar = CovarianceLayer(self.shape, channels=1)

    def forward(self, x, y):
        if x.shape[1] == 3:
            x_g = rgb_to_yiq(x)[:, :1, :, :]  # take the Y part
            y_g = rgb_to_yiq(y)[:, :1, :, :]  # take the Y part
        else:
            assert x.shape[1] == 1
            x_g = x  # take the Y part
            y_g = y  # take the Y part
        c = torch.mean(self.covar(x_g, y_g) ** 2)
        vv = torch.mean(self.var(x_g) * self.var(y_g))
        return c / vv

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x,weight_map=None):
        self.h_x = x.size()[2]
        self.w_x = x.size()[3]
        self.batch_size = x.size()[0]
        if weight_map is None:
            self.TVLoss_weight=(1, 1)
        else:
            # self.h_x = x.size()[2]
            # self.w_x = x.size()[3]
            # self.batch_size = x.size()[0]
            self.TVLoss_weight = self.compute_weight(weight_map)

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        h_tv = (self.TVLoss_weight[0]*torch.abs((x[:,:,1:,:]-x[:,:,:self.h_x-1,:]))).sum()
        w_tv = (self.TVLoss_weight[1]*torch.abs((x[:,:,:,1:]-x[:,:,:,:self.w_x-1]))).sum()
        # print(self.TVLoss_weight[0],self.TVLoss_weight[1])
        return (h_tv/count_h+w_tv/count_w)/self.batch_size


    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def compute_weight(self, img):
        gradx = torch.abs(img[:, :, 1:, :] - img[:, :, :self.h_x-1, :])
        grady = torch.abs(img[:, :, :, 1:] - img[:, :, :, :self.w_x-1])
        TVLoss_weight_x = torch.div(1,torch.exp(gradx))
        TVLoss_weight_y = torch.div(1, torch.exp(grady))

        # TVLoss_weight_x = torch.div(1, torch.abs(gradx)+0.0001)
        # TVLoss_weight_y = torch.div(1, torch.abs(grady)+0.0001)

        # TVLoss_weight_x = torch.log2(1+gradx*gradx)
        # TVLoss_weight_y = torch.log2(1+grady*grady)
        return TVLoss_weight_x, TVLoss_weight_y

