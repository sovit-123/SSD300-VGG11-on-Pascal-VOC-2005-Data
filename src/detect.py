from torchvision import transforms
from utils import rev_label_map, label_color_map
from PIL import Image, ImageDraw, ImageFont

import torch
import argparse
import cv2
import numpy as np

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a
                      match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one 
                        with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across 
                  all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or
                     you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # transform
    image = normalize(to_tensor(resize(original_image)))

    # move to default device
    image = image.to(device)

    # forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # if no objects found, the detected labels will be set to ['0.'], i.e....
    # ...['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # return original image
        return original_image

    # annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])

        # text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='black',
                  font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', 
                        default='../input/test_data/fire_fighter.jpg',
                        help='path to the test data')
    parser.add_argument('-c', '--checkpoint', 
                        default='../model_checkpoints/checkpoint_ssd300_vgg11.pth.tar', 
                        help='path to the trained model checkpoint')
    args = vars(parser.parse_args())

    # load model checkpoint
    checkpoint = args['checkpoint']
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    print(f"\nLoaded checkpoint from epoch {start_epoch}.\n")
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    # transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    img_path = args['input']
    image_name = img_path.split('/')[-1]
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    annotated_image = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
    annotated_image = np.asarray(annotated_image)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Annotated image', annotated_image)
    cv2.waitKey(0)
    cv2.imwrite(f"../outputs/{image_name}", annotated_image)