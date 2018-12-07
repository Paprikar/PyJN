"""Demo for use yolo v3
"""
import os
import argparse
import time
import cv2
import numpy as np
from model.yolo_model import YOLO

parser = argparse.ArgumentParser()
parser.add_argument(
    '-model_path',
    help='Path to converted to keras model file.')
parser.add_argument(
    '-classes_path',
    help='Path classes list file.')

def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(416, 416, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.
        classes: ndarray, classes of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {:.3f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image

if __name__ == '__main__':
    args = parser.parse_args()

    if args.model_path is not None:
        model_path = os.path.expanduser(args.model_path)
        assert model_path.endswith('.h5'), 'Model path {} is not a .h5 file'.format(model_path)
    else:
        model_path = 'data/yolo.h5'

    if args.classes_path is not None:
        classes_path = os.path.expanduser(args.classes_path)
        assert classes_path.endswith('.txt'), 'Classes path {} is not a .txt file'.format(classes_path)
    else:
        classes_path = 'data/coco_classes.txt'

    yolo = YOLO(model_path, 0.6, 0.5)
    all_classes = get_classes(classes_path)

    # detect images in test floder.
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            for f in files:
                print(f)
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image = detect_image(image, yolo, all_classes)
                cv2.imwrite('images/res/' + f, image)
