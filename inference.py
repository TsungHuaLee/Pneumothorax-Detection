# +
import sys
import os
import torch
import cv2
import getopt
import numpy as np
import segmentation_models_pytorch as smp
import ttach as tta

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# from utils import *
# from conf import *

import utils
import conf
from model import *

"""
python main.py -i "path" -o "path"
"""


def usage():
    print(' -i (str) Input image\n'
          ' -o (str) Output image\n'
          ' -g (int) Using GPU')
    
    
def record(prob_info, status_code, prob={'diseased_mask': 0.0}):
    with open(prob_info, 'w') as f:
        f.write("{:.3f},{}".format(prob['diseased_mask'], status_code))
    print('Event: {}'.format(status_code))
    
if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__))
    inputs_folder = "{}{}".format(abs_path, conf.INPUT_FOLDER)
    outputs_folder = "{}{}".format(abs_path, conf.OUTPUT_FOLDER)
    model = "{}{}".format(abs_path, conf.MODEL_PATH)
    
    isGPU = False

    try:
        options, args = getopt.getopt(sys.argv[1:],
                                      "i:o:g:",
                                      ["input=", "output=", "isGPU="])
        for opt, value in options:
            if opt in ('-i', '--input'):
                input_path = value

            elif opt in ('-o', '--output'):
                output_path = value

            elif opt in ('-g', '--isGPU'):
                isGPU = True if value == '1' else False
    except getopt.GetoptError:
        usage()
        exit(1)
    
    status_code = '0'
    
    # 0: successed, 
    # 1.1: image loading or preprocessing failed. 
    # 1.2.1: segmentation model loading failed.
    # 1.2.2: segmentation model inferencing failed.
    # 1.3.1: classification model loading failed.
    # 1.3.2: classification model inferencing failed.
    # 1.4: image save failed.
    
    image_name = input_path.split('.')[-2]
    
    input_path = '{}/{}'.format(inputs_folder, input_path)
    output_path = '{}/{}'.format(outputs_folder, image_name)
    prob_info = '{}.txt'.format(output_path)
    
    info = 'Input: {}\n'\
           'Output: {}\n' \
           'Using GPU: {}'.format(input_path, output_path, isGPU)

    DEVICE = 'cuda' if isGPU else 'cpu'
    print("Loading Model ...")
    print("Data Loading and Preprocessing...")
    try:
        '''
        file type = 0, not define
        file type = 1, image type
        file type = 2, dcm
        '''
        file_type = utils.checkFileExntension(input_path)
        if not file_type:
            status_code = '1.1'
            record(prob_info, status_code)
            exit(1)
        else:
            processed_image = utils.InferencePreprocessing(input_path, file_type, height = 512, width = 512)
    except:
        status_code = '1.1'
        record(prob_info, status_code)
        exit(1)
        
        
    print("Model Loading...")
    try:
        if isGPU:
            best_model = declare_model()
            best_model.load_state_dict(torch.load(model))
            best_model.cuda()
        else:
            best_model = declare_model()
            best_model.load_state_dict(torch.load(model))
            best_model.cpu()
    except:
        status_code = '1.2.1'
        record(prob_info, status_code)
        exit(1)

# #     # best_model = tta.SegmentationTTAWrapper(
# #     #     best_model, tta.aliases.hflip_transform(), merge_mode='mean')

    print("Model Predicting...")
    try:
        x_tensor = torch.from_numpy(processed_image).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            pr_mask = best_model(x_tensor)
        pr_mask = torch.sigmoid(pr_mask)
        pr_mask = pr_mask.squeeze().cpu().numpy()

    except:
        status_code = '1.2.2'
        record(prob_info, status_code)
        exit(1)
        
    print("Visiualizing")
    try:
        processed_image = processed_image.astype('uint8')
        prob = utils.result_visualize(image_name,
                                save_path=output_path,
                                ori_image=processed_image,
                                diseased_mask=pr_mask,
                                )
    except:
        status_code = '1.4'
        record(prob_info, status_code)
        exit(1)
    
    print("Processing Finished!")

    record(prob_info, status_code, {'diseased_mask': prob})
    
