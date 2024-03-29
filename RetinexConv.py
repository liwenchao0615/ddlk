import argparse
import os
import time
from glob import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from net.Lightnet import ReflectionLightnet, IluminationLightnet
from net.losses import ExclusionLoss, TVLoss, GradientLoss, Loss, L_exp, illumination_smooth_loss, \
    reflectance_smooth_loss
from utils.Decomposition import init_decomposition, adjust_gammma
from utils.downsampler import pair_downsampler
from utils.initInput import init_input
from utils.utils import net_init

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

noise_type = 'gauss'
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/Test', help='test image folder')
parser.add_argument('--output', type=str, default='./result', help='result folder')
parser.add_argument('--epochs', dest='epochs', type=int, default=300, help='number of epochs to train')
arg = parser.parse_args()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # 输出当前使用的设备（如：cuda:0表示第一块GPU）


def train():
    input_root = arg.input
    output_root = arg.output

    datasets = ['DICM']
    #加载数据集
    for dataset in datasets:
        input_folder = os.path.join(input_root, dataset)
        output_folder = os.path.join(output_root, dataset)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # 读取文件夹下面的所有图像
        images = glob(input_folder+'/*.*')
        images.sort()
        #遍历每一张图像
        for i in range(len(images)):
            filename = os.path.basename(images[i])
            images_path = os.path.join(input_folder, filename)
            images_path_out = os.path.join(output_folder, filename)
            image = cv2.imread(images_path)
            image_cv = cv2.resize(image,(512, 512))

            #初始化网络输入，反射分量
            image_input = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            reflection_net_input = transforms.ToTensor()(image_input).float().unsqueeze(0).to(device)

            image_temp = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            v = image_temp[:, :, 2]
            image_v_tensor = transforms.ToTensor()(v).float().unsqueeze(0).to(device)

            # adjust_illu = torch.pow(image_v_tensor,0.15)
            # image_result =  reflection_net_input/adjust_illu
            # image_result = image_result.cpu()
            # image_result = transforms.ToPILImage()(image_result.squeeze(0))
            # image_result.save('output/output/illumination_test_output.png')




            #初始化网络输入，光照分量
            illumination_net_input = torch.cat((image_v_tensor,  image_v_tensor, image_v_tensor), dim=1)
            # image_v = v[np.newaxis, :, :]
            # transform = transforms.Compose([
            #     transforms.Resize((512, 512)),
            #     transforms.ToTensor(),
            # ])
            #
            # image_v = transform(image_v)
            # image_v_tensor = image_v.unsqueeze(0)

            # image_tensor = transform(cv2.resize(image_cv, (512, 512)))
            # image_tensor = image_tensor.unsqueeze(0)
            # image_tensor = image_tensor.to(device)
            # cv2.imwrite('output/output/image_cv.png', image_cv)
            # image_pil = Image.open(images_path).convert('RGB')
            # image_pil.save('output/output/image_pil.png')

            #分解出光照分量与反射分量
            # illumination, reflection = init_decomposition(image_cv, image_pil)

            # illumination = cv2.resize(illumination, (512, 512))
            # reflection = cv2.resize(reflection, (512, 512))
            # cv2.imwrite('output/output/image_illumination.png', illumination)
            # cv2.imwrite('output/output/image_reflection.png', reflection)
            # image_cv = cv2.resize(image_cv, (512, 512))

            #初始化网络
            reflection_net = ReflectionLightnet()
            reflection_net.apply(net_init)
            reflection_net = reflection_net.to(device)
            illumination_net = IluminationLightnet()
            illumination_net.apply(net_init)
            illumination_net = illumination_net.to(device)

            #初始化网络输入
            # illumination_net_input, reflection_net_input = init_input(illumination, reflection)
            # illumination_net_input = illumination_net_input.type(torch.float32)
            # reflection_net_input = reflection_net_input.type(torch.float32)
            # illumination_tensor = illumination

            # illumination_tensor = transform(illumination_tensor).unsqueeze(0)

            # illumination_tensor = illumination_v.type(torch.float32)
            # reflection_tensor = transforms.ToTensor()(image_input).unsqueeze(0).resize_(1, 3, 512, 512).type(torch.float32)


            # illumination_tensor = illumination_tensor.to(device)
            # illumination_net_input = illumination_tensor
            # reflection_net_input = reflection_tensor.to(device)
            #保存网络输入
            illumination_input = illumination_net_input.cpu()
            illumination_input = transforms.ToPILImage()(illumination_input.squeeze(0))
            illumination_input.save('output/output/illumination_input.png')
            #保存网络输入
            reflection_input = reflection_net_input.cpu()
            reflection_input = transforms.ToPILImage()(reflection_input.squeeze(0))
            reflection_input.save('output/output/reflection_input.png')


            #下采样
            illumination_down1, illumination_down2 = pair_downsampler(illumination_net_input)
            reflection_down1, reflection_down2 = pair_downsampler(reflection_net_input)
            illumination_down1 = illumination_down1.to(device)
            illumination_down2 = illumination_down2.to(device)
            reflection_down1 = reflection_down1.to(device)
            reflection_down2 = reflection_down2.to(device)

            #定义损失函数
            l1_loss = nn.SmoothL1Loss().to(device)
            mse_loss = nn.MSELoss().to(device)
            exclusion_loss = ExclusionLoss().to(device)
            tv_loss = TVLoss().to(device)
            gradient_loss = GradientLoss().to(device)
            convloss = Loss().to(device)
            lossexp = L_exp(16)

            #定义优化器
            parameters = [p for p in reflection_net.parameters()] + \
                         [p for p in illumination_net.parameters()]
            optimizer = torch.optim.Adam(parameters, lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
            print("Processing: {}".format(images_path_out.split("/")[-1]))
            start = time.time()
            #开始训练
            for j in range(arg.epochs):
                optimizer.zero_grad()
                #计算网络输出
                illumination_out = illumination_net(illumination_net_input)

                illumination_output = torch.clamp(illumination_out, 0, 1)
                #对输出结果进行保存,pli库
                if (j+1) % 10 == 0:
                    res_illumination = illumination_output.cpu()
                    illumination_img = transforms.ToPILImage()(res_illumination.squeeze(0))
                    illumination_img.save('output/illumination/illumination_output{:d}.png'.format(j+1))

                illumination_down1_out = illumination_net(illumination_down1)
                illumination_down2_out = illumination_net(illumination_down2)

                reflection_out = reflection_net(reflection_net_input)
                # 对输出结果进行保存，pil库
                if (j+1) % 10 == 0:
                    reflection_out = torch.clamp(reflection_out, 0, 1)
                    res_reflection = reflection_out.cpu()
                    reflection_img = transforms.ToPILImage()(res_reflection.squeeze(0))
                    reflection_img.save('output/reflection/reflection_output{:d}.png'.format(j+1))

                reflection_down1_out = reflection_net(reflection_down1)
                reflection_down2_out = reflection_net(reflection_down2)

                # # 对输出结果进行保存，cv2库
                # illumination_outputcopy = illumination_out.cpu()
                # illumination_outputcopy = illumination_outputcopy.squeeze(0).detach().numpy()
                # illumination_outputcopy = illumination_outputcopy.transpose(1, 2, 0)
                # illumination_outputcopy = np.clip(illumination_outputcopy, 0, 255).astype(np.uint8)
                # cv2.imwrite('output/output/illumination_output.png', illumination_outputcopy)
                #
                # # 对输出结果进行保存，cv2库
                # reflection_outputcopy = reflection_out.cpu()
                # reflection_outputcopy = reflection_outputcopy.squeeze(0).detach().numpy()
                # reflection_outputcopy = reflection_outputcopy.transpose(1, 2, 0)
                # reflection_outputcopy = np.clip(reflection_outputcopy*255, 0, 255).astype(np.uint8)
                # cv2.imwrite('output/output/reflection_output.png', reflection_outputcopy)

                #计算损失
                total_loss = 0.5*tv_loss(illumination_out, reflection_out)
                total_loss += 0.0001*tv_loss(reflection_out)
                total_loss += convloss(illumination_down1, illumination_down2, illumination_out, illumination_down1_out, illumination_down2_out)
                total_loss += convloss(reflection_down1, reflection_down2, reflection_out, reflection_down1_out, reflection_down2_out)
                # total_loss += l1_loss(illumination_out, illumination_net_input)
                total_loss += mse_loss(reflection_out*illumination_out, reflection_net_input)
                # total_loss += illumination_smooth_loss(illumination_net_input, illumination_out)
                # total_loss += reflectance_smooth_loss(reflection_net_input, illumination_out, reflection_out)
                # total_loss += 10*torch.mean(lossexp(reflection_out, 0.6))
                # total_loss += 10 * torch.mean(lossexp(illumination_out, 0.5))
                # illumination_1 = np.max(illumination_outputcopy, axis=2)
                # illumination_1 = np.clip(np.asarray([illumination_1 for _ in range(3)]), 0.000002, 255)
                # illumination_1 = np.clip(adjust_gammma(illumination_1), 1, 255).astype(np.uint8)

                adjust_illu = torch.pow(illumination_output, 0.15)

                if (j+1) % 10 == 0:
                    image_result_adjust_illu = adjust_illu.cpu()
                    image_result_adjust_illu = transforms.ToPILImage()(image_result_adjust_illu.squeeze(0))
                    image_result_adjust_illu.save('output/output/image_adjust_illu{:d}.png'.format(j+1))

                # image_result = reflection_out*adjust_illu
                # image_result = image_result / adjust_illu
                image_result = reflection_out * illumination_out
                image_result = image_result / adjust_illu
                image_result = torch.clamp(image_result, 0, 1)
                if (j+1) % 10 == 0:
                    image_result1 = image_result.cpu()
                    image_result1 = transforms.ToPILImage()(image_result1.squeeze(0))
                    image_result1.save('output/image_result{:d}.png'.format(j+1))
                # image_result = image_result.squeeze(0).detach().numpy()
                # best_result = np.clip((image_result / illumination_1)*255, 1, 255)
                # best_result = best_result.transpose(1, 2, 0)
                # cv2.imwrite('output/output/best_result.png', best_result)

                # image_result = image_result.transpose(1, 2, 0)
                # image_result = np.clip(image_result, 0, 255).astype(np.uint8)
                # cv2.imwrite('output/output/image_result.png', image_result)
                if j == 299:
                    image_result2 = image_result.cpu()
                    image_result2 = transforms.ToPILImage()(image_result2.squeeze(0))
                    image_result2.save('result/bestresult{:d}.png'.format(i+1))
                print('Iteration {:5d}    Loss {:5f}'.format(j+1, total_loss.item()))
                total_loss.backward()
                optimizer.step()
                scheduler.step()
            end = time.time()
            print("time:%.4f" % (end - start))

if __name__ == "__main__":
    train()