# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:18:49 2024

@author: Tuan
"""

import torch
from torchvision import transforms
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from ADJSCCV_model import ADJSCC_V
from data_loader import ImgData

import cycleGAN_model as cycleGAN
from CNNclassifier_model import DigitClassifierCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join('.', 'model')
classifier_path = os.path.join(model_path, 'classifier')
GAN_path = os.path.join(model_path, 'cycleGAN')

DS_NAME = ["MNIST", "MNISTM", "SYN", "USPS"]

BATCH_SIZE = 64

CHANNEL = 'AWGN'  # Choose AWGN or Fading
N_CHANNELS = 256
KERNEL_SIZE = 5

IMAGE_SIZE = 32

enc_out_shape = [48, IMAGE_SIZE//4, IMAGE_SIZE//4]

# criterion = nn.CrossEntropyLoss()

normalize_func = transforms.Normalize([0.5], [0.5])

pair_combination = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])

eval_info = []

for src_domain, dst_domain in pair_combination:

    current_setting = f'cycleGAN_{DS_NAME[src_domain]}_to_{DS_NAME[dst_domain]}'

    print("\n=================================================================")
    print(f'Current setting: {current_setting}')

    classifier_name = f'{DS_NAME[dst_domain]}_classifier.pkl'
    classifier = DigitClassifierCNN()
    classifier.load_state_dict(torch.load(os.path.join(classifier_path, classifier_name)))
    classifier.to(device).eval()

    GAN_name = f'gen_{DS_NAME[src_domain]}_{DS_NAME[dst_domain]}.pkl'
    GAN = cycleGAN.Generator(in_channels=3, out_channels=3, conv_dim=12)
    GAN.load_state_dict(torch.load(os.path.join(GAN_path, GAN_name)))
    GAN.to(device).eval()

    ADJSCCV_path = os.path.join(model_path, 'ADJSCCV', f'{DS_NAME[dst_domain]}')

    for epoch in np.arange(5, 51, 5):

        dataset = ImgData(f'./dataset/{DS_NAME[src_domain]}_test.pt', IMAGE_SIZE, IMAGE_SIZE)
        ds_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False)

        ADJSCCV_name = f'JSCC-V_{DS_NAME[dst_domain]}_epoch_{epoch}.pt'
        ADJSCCV = ADJSCC_V(enc_out_shape, KERNEL_SIZE, N_CHANNELS)
        ADJSCCV.load_state_dict(torch.load(os.path.join(ADJSCCV_path, ADJSCCV_name)))
        ADJSCCV.to(device).eval()

        correct_pred = 0

        for _, (src_img, target_label) in tqdm(enumerate(ds_loader), total=len(ds_loader), desc=f'Epoch {epoch}'):

            src_img, target_label = src_img.to(device), target_label.to(device)

            DA_img = GAN(src_img)

            SNR_TEST = torch.randint(0, 28, (DA_img.shape[0], 1)).cuda()
            CR = 0.1+0.9*torch.rand(DA_img.shape[0], 1).cuda()

            DA_img_rec = normalize_func(ADJSCCV(DA_img, SNR_TEST, CR, CHANNEL))

            prediction = classifier(DA_img_rec)
            prediction = prediction.argmax(dim=1, keepdim=True)
            correct_pred += prediction.eq(target_label.view_as(prediction)).sum().item()

        accuracy = correct_pred / len(dataset)

        print("-----------------------------------------------------------------")
        print(f'Correctly classified {correct_pred} out of {len(dataset)} samples.')
        print(f'Accuracy = {accuracy:.4f}')
        print("-----------------------------------------------------------------")

        eval_info.append(['cycleGAN',
                          f'{DS_NAME[src_domain]}',
                          f'{DS_NAME[dst_domain]}',
                          epoch,
                          accuracy])

    print("=================================================================")


    # Reverse the order

    src_domain, dst_domain = dst_domain, src_domain

    current_setting = f'cycleGAN_{DS_NAME[src_domain]}_to_{DS_NAME[dst_domain]}'

    print("\n=================================================================")
    print(f'Current setting: {current_setting}')

    classifier_name = f'{DS_NAME[dst_domain]}_classifier.pkl'
    classifier = DigitClassifierCNN()
    classifier.load_state_dict(torch.load(os.path.join(classifier_path, classifier_name)))
    classifier.to(device).eval()

    GAN_name = f'gen_{DS_NAME[src_domain]}_{DS_NAME[dst_domain]}.pkl'
    GAN = cycleGAN.Generator(in_channels=3, out_channels=3, conv_dim=12)
    GAN.load_state_dict(torch.load(os.path.join(GAN_path, GAN_name)))
    GAN.to(device).eval()

    ADJSCCV_path = os.path.join(model_path, 'ADJSCCV', f'{DS_NAME[dst_domain]}')

    for epoch in np.arange(5, 51, 5):

        dataset = ImgData(f'./dataset/{DS_NAME[src_domain]}_test.pt', IMAGE_SIZE, IMAGE_SIZE)
        ds_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False)

        ADJSCCV_name = f'JSCC-V_{DS_NAME[dst_domain]}_epoch_{epoch}.pt'
        ADJSCCV = ADJSCC_V(enc_out_shape, KERNEL_SIZE, N_CHANNELS)
        ADJSCCV.load_state_dict(torch.load(os.path.join(ADJSCCV_path, ADJSCCV_name)))
        ADJSCCV.to(device).eval()

        correct_pred = 0

        for _, (src_img, target_label) in tqdm(enumerate(ds_loader), total=len(ds_loader), desc=f'Epoch {epoch}'):

            src_img, target_label = src_img.to(device), target_label.to(device)

            DA_img = GAN(src_img)

            SNR_TEST = torch.randint(0, 28, (DA_img.shape[0], 1)).cuda()
            CR = 0.1+0.9*torch.rand(DA_img.shape[0], 1).cuda()

            DA_img_rec = normalize_func(ADJSCCV(DA_img, SNR_TEST, CR, CHANNEL))

            prediction = classifier(DA_img_rec)
            prediction = prediction.argmax(dim=1, keepdim=True)
            correct_pred += prediction.eq(target_label.view_as(prediction)).sum().item()

        accuracy = correct_pred / len(dataset)

        print("-----------------------------------------------------------------")
        print(f'Correctly classified {correct_pred} out of {len(dataset)} samples.')
        print(f'Accuracy = {accuracy:.4f}')
        print("-----------------------------------------------------------------")

        eval_info.append(['cycleGAN',
                          f'{DS_NAME[src_domain]}',
                          f'{DS_NAME[dst_domain]}',
                          epoch,
                          accuracy])

    print("=================================================================")



df = pd.DataFrame(eval_info, columns=['GAN Network',
                                      'Source Domain',
                                      'Destination Domain',
                                      'Epoch',
                                      'Accuracy'])

eval_info_path = os.path.join('.', 'eval_info')
os.makedirs(eval_info_path, exist_ok=True)
eval_info_path = os.path.join(eval_info_path, 'eval_info_cycleGAN.csv')
df.to_csv(eval_info_path, index=True)

print(f'Saving eval info to {eval_info_path}')
print()

print('All done!')































