import os
import numpy as np
import cv2
import configargparse



def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='/home/geratsbga1/Surgical-OmniMotion/omnimotion/out')
    parser.add_argument('--exp_name', type=str, default='video17_01803')
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--data_path', type=str, default='/home/geratsbga1/CholecSeg8k')
    parser.add_argument('--mask_path', type=str, default='video17/video17_01803/_omnimotion/mask/Grasper')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Read arguments
    args = config_parser()
    
    # Read point tracks
    points_file = os.path.join(args.out_path, args.exp_name, 'vis',
                               f'_omnimotion_{args.iters:06d}_foreground_0.npy')
    points = np.load(points_file)[..., :2]
    points = np.rint(points).astype(int)
    num_points = points.shape[1]
    
    # Per frame, check ratio of points that are still within tool mask
    accuracies = []
    num_frames = len(points)
    for i in range(1, len(points)):
        
        # Read mask image (alpha channel)
        mask_file = os.path.join(args.data_path, args.mask_path, f'{i:05d}.png')
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)[..., 3]
        
        # Count number of correct points
        correct = 0
        for point in points[i]:
            if mask[point[1], point[0]] > 0:
                correct += 1
                
        # Calculate accuracy
        accuracy = float(correct) / float(num_points)
        accuracies.append(accuracy)
        
    # Print and store accuracies
    print(f'Num frames: {num_frames} | Num points: {num_points} | Avg acc: {np.mean(accuracies):.3f}')
    print('\n', accuracies, '\n')
    tool_type = args.mask_path.split('/')[-1]
    save_file = os.path.join(args.out_path, args.exp_name, 'vis', f'tool_tracking_accuracy_{tool_type}.npy')
    np.save(save_file, accuracies)