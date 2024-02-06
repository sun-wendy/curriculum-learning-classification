import os
import argparse
from imagecorruptions import corrupt
import cv2 as cv


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--corruptions',
        type=str,
        nargs='+',
        default='benchmark',
        choices=[
            'all', 'benchmark', 'noise', 'blur', 'weather', 'digital',
            'holdout', 'None', 'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'motion_blur', 'zoom_blur', 'snow',
            'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
            'pixelate', 'jpeg_compression', 'speckle_noise',
            'spatter', 'saturate'
        ],
        help='corruptions')
    parser.add_argument(
        '--severity',
        type=int,
        default=3,
        help='corruption severity level')
    args = parser.parse_args()
    return args


def save_augmented(corruptions, severity):
    if 'all' in corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression', 'speckle_noise', 'spatter', 'saturate'
        ]
    elif 'benchmark' in corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
    elif 'noise' in corruptions:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise']
    elif 'blur' in corruptions:
        corruptions = [
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
        ]
    elif 'weather' in corruptions:
        corruptions = ['snow', 'frost', 'fog', 'brightness']
    elif 'digital' in corruptions:
        corruptions = [
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
    elif 'holdout' in corruptions:
        corruptions = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    elif 'None' in corruptions:
        corruptions = ['None']
        severity = 0
    else:
        corruptions = corruptions
    
    data_folder = './data/composite/test'
    corrupted_data_folder = './data/corrupted_composite/test'

    for corruption in corruptions:
            for label in os.listdir(data_folder):
                for img_name in os.listdir(os.path.join(data_folder, label)):
                    img = cv.imread(os.path.join(data_folder, label, img_name))
                    corrupted_image = corrupt(
                        img,
                        corruption_name=corruption,
                        severity=severity
                    )
                    if not os.path.exists(os.path.join(corrupted_data_folder, label)):
                        os.makedirs(os.path.join(corrupted_data_folder, label))
                    cv.imwrite(
                        os.path.join(corrupted_data_folder, label, f'{img_name}_{corruption}.png'),
                        corrupted_image
                    )


if __name__ == "__main__":
    args = parse_args()
    save_augmented(args.corruptions, args.severity)
