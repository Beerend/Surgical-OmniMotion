import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='/data/SCARED', type=str)
    parser.add_argument('--results_path', default='/app/Surgical-OmniMotion/results', type=str)
    parser.add_argument('--dataset_num', default=1, type=int)
    parser.add_argument('--keyframe_num', default=1, type=int)
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--delta_power', default=1, type=int)
    args = parser.parse_args()
    return args


def read_pred_depths(results_path, exp_name):
    exp_path = os.path.join(results_path, exp_name)
    pred_depths = np.load(os.path.join(exp_path, 'depth', 'data_020000.npy'))
    pred_depths = pred_depths.astype(np.float32)
    return pred_depths


def read_target_depths(keyframe_path, W=None, H=None):
    depth_path = os.path.join(keyframe_path, 'data', 'depthmap')
    depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith('.png')])

    target_depths = []
    for file in depth_files:
        img = cv.imread(os.path.join(depth_path, file))
        if W is not None and H is not None:
            img = cv.resize(img, (W, H), interpolation=cv.INTER_NEAREST)
        img = img[:, :, 0]
        img = img.astype(np.float32)
        target_depths.append(img)
    target_depths = np.stack(target_depths)
    return target_depths


def read_color_images(keyframe_path, W=None, H=None):
    color_path = os.path.join(keyframe_path, 'data', 'color')
    color_files = sorted([f for f in os.listdir(color_path) if f.endswith('.png')])

    color_imgs = []
    for file in color_files:
        img = cv.imread(os.path.join(color_path, file))
        if W is not None and H is not None:
            img = cv.resize(img, (W, H), interpolation=cv.INTER_NEAREST)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        color_imgs.append(img)
    color_imgs = np.stack(color_imgs)
    return color_imgs


def print_statistics(data, title):
    print(f'> {title}: {data.shape} (shape) {np.mean(data):.3f} (mean) {np.std(data):.3f} (std)')


def rescale_lstsq_np(imgs, targets):
    rescaled_imgs = []
    for img, target in zip(imgs, targets):
        nonzero_idxs = np.nonzero(target)
        nonzero_target = target[nonzero_idxs].flatten()
        nonzero_img = img[nonzero_idxs].flatten()
        nonzero_img = np.vstack([nonzero_img, np.ones(len(nonzero_img))]).T
        gain, offset = np.linalg.lstsq(nonzero_img, nonzero_target, rcond=None)[0]
        img = img * gain + offset
        img = np.clip(img, a_min=0.0, a_max=None)
        rescaled_imgs.append(img)
    rescaled_imgs = np.stack(rescaled_imgs)
    return rescaled_imgs


def calc_mse(preds, targets):
    errors = (preds - targets)**2
    target_mask = np.where(targets==0.0, 0.0, 1.0)
    errors = errors * target_mask
    error = np.sum(errors) / np.sum(target_mask)
    return error


def calc_mae(preds, targets):
    errors = np.abs(preds - targets)
    target_mask = np.where(targets==0.0, 0.0, 1.0)
    errors = errors * target_mask
    error = np.sum(errors) / np.sum(target_mask)
    return error


def calc_absrel(preds, targets):
    errors = np.abs(preds - targets)
    target_mask = np.where(targets==0.0, 0.0, 1.0)
    errors = errors * target_mask
    errors = np.divide(errors, targets, out=np.zeros_like(errors), where=targets!=0.0)
    error = np.sum(errors) / np.sum(target_mask)
    return error


def calc_delta_threshold(preds, targets, delta_power=1):
    pred_div_targ = np.divide(preds, targets, out=np.zeros_like(preds), where=targets!=0.0)
    targ_div_pred = np.divide(targets, preds, out=np.zeros_like(targets), where=preds!=0.0)
    errors = np.max(np.stack([pred_div_targ, targ_div_pred]), axis=0)
    target_mask = np.where(targets==0.0, 0.0, 1.0)
    thr_errors = np.where(errors > 1.25**delta_power, 1.0, 0.0)
    thr_errors = thr_errors * target_mask
    error = np.sum(thr_errors) / np.sum(target_mask)
    return error

if __name__ == '__main__':
    args = config_parser()
    
    print('Reading depth predictions...')
    pred_depths = read_pred_depths(args.results_path, args.exp_name)
    H, W = pred_depths[0].shape
    
    print('Reading depth targets...')
    keyframe_path = os.path.join(args.dataset_path, f'dataset_{args.dataset_num}',
                                 f'keyframe_{args.keyframe_num}')
    target_depths = read_target_depths(keyframe_path, W=W, H=H)
    color_imgs = read_color_images(keyframe_path)
    
    print('Some statistics:')
    print_statistics(pred_depths, 'Predictions')
    print_statistics(target_depths, 'Targets')
    
    print('Reshift and rescale predictions with Least Squares...')
    rescaled_pred_depths = rescale_lstsq_np(pred_depths, target_depths)
    
    print('Results:')
    mse = calc_mse(rescaled_pred_depths, target_depths)
    mae = calc_mae(rescaled_pred_depths, target_depths)
    absrel = calc_absrel(rescaled_pred_depths, target_depths)
    delta = calc_delta_threshold(rescaled_pred_depths, target_depths,
                                 delta_power=args.delta_power)
    print(f'> MSE: {mse:.2f} mm^2')
    print(f'> MAE: {mae:.2f} mm')
    print(f'> AbsRel: {absrel*100.0:.2f} %')
    print(f'> Delta^{args.delta_power}: {delta*100.0:.2f} %')