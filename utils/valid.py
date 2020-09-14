# +
import os
import albumentations as albu
import cv2
import numpy as np
import pydicom

def checkFileExntension(input_path):
    img_type = ['.png', '.jpeg', '.jpg']
    filename, file_extension = os.path.splitext(input_path)
    if file_extension in img_type:
        return 1
    elif file_extension == '.dcm':
        return 2
    else:
        return 0
    
def image_mask_preprocessing(image, mask, height = 512, width = 512, **kwargs):
    larger_side = max(image.shape[0], image.shape[1])
    # padding to square and resize
    aug = albu.Compose([
        albu.PadIfNeeded(min_height=larger_side, min_width=larger_side, always_apply=True, border_mode=0),
        albu.Resize(height=height, width=width , always_apply=True,)
    ])
    
    sample = aug(image=image, mask=mask)
    image, mask = sample['image'], sample['mask']
    
    # normalize
    if 'is_norm' in kwargs and kwargs['is_norm'] == False:
        pass
    else:
        image = (image - image.min()) / (image.max() - image.min()) * (255 - 0) + 0
        image = image.astype('uint8')
    
    mask = np.where(mask > 0, 1, 0)
    # convert to 3 channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, mask

def InferencePreprocessing(input_path, file_type = 1, **kwargs):
    # file_type 1 : image type
    # file_type 2 : dcm
    if file_type == 1:
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    elif file_type == 2:
        dcm = pydicom.dcmread(input_path)
        image = dcm.pixel_array
        
    processed_image, processed_mask = image_mask_preprocessing(image, np.zeros_like(image), **kwargs)
    
    # apply preprocessing
    processed_image = processed_image.transpose(2, 0, 1).astype('float32')
    return processed_image

def result_visualize(image_name, save_path, ori_image, diseased_mask):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    cmap_no_background = LinearSegmentedColormap.from_list("", ["none", "blue", 'cyan', 'green', 'orange', 'red'])

    ori_image = ori_image.transpose(1,2,0)
    plt.figure(figsize=(8, 6))
    plt.imshow(ori_image, cmap = "gray")
    plt.axis('off')
    _save_path = "{0}_ori.png".format(save_path)
    plt.savefig(_save_path, bbox_inches='tight') 

    plt.figure(figsize=(8, 6))
    plt.imshow(ori_image, cmap = "gray")
    plt.imshow(diseased_mask.squeeze(), alpha=0.5, cmap= cmap_no_background, vmin=0, vmax=1)
    plt.colorbar()
    plt.axis('off')
    _save_path = "{0}_pred.png".format(save_path)
    plt.savefig(_save_path, bbox_inches='tight')
    
    return True

def ClsInferencePreprocessing(image, mask):
    '''
    Args:
        image: A numpy array with shape (C, H, W). image is the input of segmentation model
        mask: A numpy array with shape (H, W). mask is the output of segmentation model
    Returns:
        new_image: concatenate image and mask, shape == (4, H, W)
    Raises:
        AssertError: image dtype differ from mask dtype
    '''
    image = image.transpose(1,2,0)
    assert image.dtype == mask.dtype
    if len(mask.shape) > 2:
        mask = mask.transpose(1,2,0)
    else:
        mask = np.expand_dims(mask, axis=-1)
        
    new_image = np.concatenate((image, mask), axis = -1)
    
    new_image = new_image.transpose(2, 0, 1).astype('float32')
    
    return new_image
