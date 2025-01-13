import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import configargparse


color_class_mapping = {381: 1,
                       490: 2,
                       483: 3,
                       457: 4,
                       444: 5,
                       425: 6,
                       340: 7,
                       255: 8,
                       510: 9,
                       608: 10,
                       580: 11,
                       178: 12,
                       185: 13}

class_name_mapping = {1: 'Black Background',
                      2: 'Abdominal Wall',
                      3: 'Liver',
                      4: 'Gastrointestinal Tract',
                      5: 'Fat',
                      6: 'Grasper',
                      7: 'Connective Tissue',
                      8: 'Blood',
                      9: 'Cystic Duct',
                      10: 'L-hook Electrocautery',
                      11: 'Gallbladder',
                      12: 'Hepatic Vein',
                      13: 'Liver Ligament'}


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/geratsbga1/CholecSeg8k')
    parser.add_argument('--vid_num', type=int, default=17)
    parser.add_argument('--frame_num', type=int, default=1803)
    parser.add_argument('--mask_ids', type=int, default=[6, 10, 11], nargs='*')
    parser.add_argument('--img_size', type=str, default='quant')
    args = parser.parse_args()
    return args


def crop_pixels_vertical(imgs, W_new=None):
    H, W = imgs[0].shape[:2]
    if W_new is None:
        crop_pixels = (W-H)//2
    else:
        crop_pixels = (W-W_new)//2
    imgs_new = []
    for img in imgs:
        imgs_new.append(img[:, crop_pixels:-crop_pixels])
    imgs = np.stack(imgs_new)
    return imgs


def resize_imgs(imgs, H, W):
    imgs_new = []
    for img in imgs:
        imgs_new.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST))
    imgs = np.stack(imgs_new)
    return imgs


def read_color_imgs(path):
    files = [f for f in os.listdir(path) if f.endswith('endo.png')]
    files.sort()
    imgs = []
    for file in files:
        imgs.append(cv2.imread(os.path.join(path, file)))
    imgs = np.stack(imgs)
    return imgs


def read_mask_imgs(path):
    files = [f for f in os.listdir(path) if f.endswith('color_mask.png')]
    files.sort()
    imgs = []
    for file in files:
        imgs.append(cv2.imread(os.path.join(path, file), -1))
    imgs = np.stack(imgs, dtype=np.int16)
    return imgs


def read_watershed_mask_imgs(path):
    files = [f for f in os.listdir(path) if f.endswith('watershed_mask.png')]
    files.sort()
    imgs = []
    for file in files:
        imgs.append(cv2.imread(os.path.join(path, file), -1))
    imgs = np.stack(imgs, dtype=np.int16)
    return imgs


def map_color_masks(masks):
    masks = np.sum(masks, axis=3)
    mapped_masks = []
    for key in color_class_mapping.keys():
        binary_mask = np.where(masks == key, color_class_mapping[key], 0)
        mapped_masks.append(binary_mask)
    masks = np.sum(np.stack(mapped_masks), axis=0)
    return masks


def seperate_masks(masks, mask_ids):
    M, H, W = masks.shape
    I = len(mask_ids)
    masks_new = np.zeros((I, M, H, W), dtype=np.float32)
    for i, mask_id in enumerate(mask_ids):
        binary_mask = np.where(masks == mask_id, 1., 0.)
        masks_new[i] = binary_mask
    masks = np.stack(masks_new)
    return masks


def create_color_masks(imgs, masks):
    imgs_new = []
    for img, mask in zip(imgs, masks):
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = mask * 255
        imgs_new.append(img_rgba)
    imgs = np.stack(imgs_new)
    return imgs


def save_imgs(imgs, path, img_names=None):
    if not os.path.isdir(path):
        os.makedirs(path)
    if img_names is None:
        img_names = np.arange(len(imgs))
        img_names = [f'{n:05d}' for n in img_names]
    for img, name in zip(imgs, img_names):
        save_name = os.path.join(path, name + '.png')
        cv2.imwrite(save_name, img)


if __name__ == '__main__':
    # Read arguments
    args = config_parser()
    
    # Create paths
    video_dir = f'video{args.vid_num:02d}'
    scene_dir = f'{video_dir}_{args.frame_num:05d}'
    input_path = os.path.join(args.path, video_dir, scene_dir)
    output_path = os.path.join(input_path, '_omnimotion')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        
    # Read RGB images and resize
    print('Read RGB images...')
    imgs = read_color_imgs(input_path)
    if args.img_size == 'quant':
        imgs = crop_pixels_vertical(imgs)
        imgs = resize_imgs(imgs, 256, 256)
    elif args.img_size == 'quali':
        imgs = crop_pixels_vertical(imgs, 640)
        
    # Save RGB images
    print('Save RGB images...')
    save_imgs(imgs, os.path.join(output_path, 'color'))
    
    # Read segmentation annotations and resize
    print('Read segmentation masks...')
    masks = read_mask_imgs(input_path)
    masks = map_color_masks(masks)
    masks = seperate_masks(masks, args.mask_ids)
    num_masks = len(args.mask_ids)
    if args.img_size == 'quant':
        masks = masks.reshape((num_masks * 80, 480, 854))
        masks = crop_pixels_vertical(masks)
        masks = resize_imgs(masks, 256, 256)
        masks = masks.reshape((num_masks, 80, 256, 256))
    elif args.img_size == 'quali':
        masks = masks.reshape((num_masks * 80, 480, 854))
        masks = crop_pixels_vertical(masks, 640)
        masks = masks.reshape((num_masks, 80, 480, 640))
        
    # Create separate mask images and save them
    for i in range(num_masks):
        class_name = class_name_mapping[args.mask_ids[i]]
        print(f'Create masks for {class_name} ({i+1}/{num_masks})...')
        if np.sum(masks[i]) == 0:
            print('> Tool or anatomy not present in video clip. Skipping...')
            continue
        color_masks = create_color_masks(imgs, masks[i])
        save_imgs(color_masks, os.path.join(output_path, 'mask', class_name))