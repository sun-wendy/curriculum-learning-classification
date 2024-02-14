import os
import cv2 as cv
import numpy as np

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def extract_classwise_instances(samples, output_dir, label_field, ext=".png"):
    print("Extracting object instances...")
    for sample in samples.iter_samples(progress=True):
        img = cv.imread(sample.filepath)
        img_h, img_w, c = img.shape
        for det in sample[label_field].detections:
            mask = det.mask
            [x, y, w, h] = det.bounding_box
            x = int(x * img_w)
            y = int(y * img_h)
            h, w = mask.shape
            mask_img = img[y:y+h, x:x+w, :]
            alpha = mask.astype(np.uint8)*255
            alpha = np.expand_dims(alpha, 2)
            mask_img = np.concatenate((mask_img, alpha), axis=2)

            label = det.label
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
            output_filepath = os.path.join(label_dir, det.id+ext)
            if mask_img is not None and mask_img.size > 0:
                cv.imwrite(output_filepath, mask_img)


def save_composite(samples, output_dir, label_field, ext=".png"):
    print("Saving composite images...")
    for sample in samples.iter_samples(progress=True):
        img = cv.imread(sample.filepath)
        img_h, img_w, c = img.shape
        output_filepath = output_dir

        counter = 0
        for i, det in enumerate(sample[label_field].detections):
            if counter > 0:
              break
            label = det.label
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
            output_filepath = os.path.join(label_dir, det.id+ext)
        cv.imwrite(output_filepath, img)


if __name__ == "__main__":
    label_field = "ground_truth"

    train_dataset = foz.load_zoo_dataset("coco-2017",
                                        split="train",
                                        label_types=["segmentations"],
                                        # max_samples=50,
                                        shuffle=True,
                                        label_field=label_field)
    print(len(train_dataset))

    test_dataset = foz.load_zoo_dataset("coco-2017",
                                        split="validation",
                                        label_types=["segmentations"],
                                        # max_samples=50,
                                        shuffle=True,
                                        label_field=label_field)
    print(len(test_dataset))

    classes = train_dataset.distinct("ground_truth.detections.label")
    print("Classes: ", classes)

    train_view = train_dataset.filter_labels(label_field, F("label").is_in(classes))
    print(train_view)
    test_view = test_dataset.filter_labels(label_field, F("label").is_in(classes))
    print(test_view)

    foreground_train_output_dir = os.path.join(os.getenv('DATASET_DIR'), 'data/foreground/train')
    foreground_test_output_dir = os.path.join(os.getenv('DATASET_DIR'), 'data/foreground/test')
    composite_train_output_dir = os.path.join(os.getenv('DATASET_DIR'), 'data/composite/train')
    composite_test_output_dir = os.path.join(os.getenv('DATASET_DIR'), 'data/composite/test')

    os.makedirs(foreground_train_output_dir, exist_ok=True)
    os.makedirs(foreground_test_output_dir, exist_ok=True)
    os.makedirs(composite_train_output_dir, exist_ok=True)
    os.makedirs(composite_test_output_dir, exist_ok=True)

    extract_classwise_instances(train_view, foreground_train_output_dir, label_field)
    extract_classwise_instances(test_view, foreground_test_output_dir, label_field)
    save_composite(train_view, composite_train_output_dir, label_field)
    save_composite(test_view, composite_test_output_dir, label_field)
