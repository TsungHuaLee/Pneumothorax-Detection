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

import conf
import utils.valid as utils
import models
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
    print(info)
    DEVICE = 'cuda' if isGPU else 'cpu'
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
    except Exception as e:
        status_code = '1.1'
        record(prob_info, status_code)
        print(e)
        exit(1)
        
        
        
    print("Model Loading...")
    try:
        if isGPU:
            best_model = models.ResPoolNet.declare_model()
            best_model.load_state_dict(torch.load(model))
            best_model.cuda()
        else:
            best_model = models.ResPoolNet.declare_model()
            best_model.load_state_dict(torch.load(model))
            best_model.cpu()
    except Exception as e:
        status_code = '1.2.1'
        record(prob_info, status_code)
        print(e)
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

    except Exception as e:
        status_code = '1.2.2'
        record(prob_info, status_code)
        print(e)
        exit(1)

    '''
        classification model
        the probability is max of segmentation output
    '''
    prob = pr_mask.max()    
    if prob >= conf.CLS_THRESHOLD:
        cls_model = "{}{}".format(abs_path, conf.CLSMODEL_PATH)
        print("Loading Classification Model")
        try:
            if isGPU:
                best_model = models.res2net.declare_model()
                best_model.load_state_dict(torch.load(cls_model))
                best_model.cuda()
            else:
                best_model = models.res2net.declare_model()
                best_model.load_state_dict(torch.load(cls_model))
                best_model.cpu()
        except Exception as e:
            status_code = '1.3.1'
            record(prob_info, status_code)
            print(e)
            exit(1)
            
        '''concat_image: (4, H, W)'''
        print("Classification Model Predicting...")
        try:
            concat_image = utils.ClsInferencePreprocessing(processed_image, pr_mask)
            with torch.no_grad():
                x_tensor = torch.from_numpy(concat_image).to(DEVICE).unsqueeze(0)
                cls = best_model(x_tensor)
            cls = torch.softmax(cls, dim=1)
            cls = cls.detach().cpu().numpy()
            cls = np.argmax(cls, axis=1)
        except Exception as e:
            status_code = '1.3.2'
            record(prob_info, status_code)
            print(e)
            exit(1)
        
        if cls == 0:
            pr_mask = pr_mask * 0.1
            
            
    print("Visiualizing")
    try:
        processed_image = processed_image.astype('uint8')
        utils.result_visualize(image_name,
                        save_path=output_path,
                        ori_image=processed_image,
                        diseased_mask=pr_mask,
                        )
    except Exception as e:
        status_code = '1.4'
        record(prob_info, status_code)
        print(e)
        exit(1)

    print("Processing Finished!")

    record(prob_info, status_code, {'diseased_mask': prob})
    
