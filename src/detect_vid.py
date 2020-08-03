from torchvision import transforms
from utils import rev_label_map, label_color_map
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

import argparse
import cv2
import numpy as np
import time
import torch

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Computation Device: ', device)

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered
                      a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one 
                        with the lower score is not suppressed via Non-Maximum
                        Suppression (NMS)
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
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, 
                                                             predicted_scores, 
                                                             min_score=min_score,
                                                             max_overlap=max_overlap, 
                                                             top_k=top_k)

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
    parser.add_argument('-i', '--input', default='../../input/test_data/video1.mp4', help='path to video')
    parser.add_argument('-c', '--checkpoint', default='../model_checkpoints/checkpoint_ssd300_vgg11.pth.tar', 
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

    cap = cv2.VideoCapture(args['input'])

    if (cap.isOpened() == False):
        print('Error while trying to read video. Plese check again...')

    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if frame_width > 800:
        frame_width = 640
        frame_height = 400

    # define codec and create VideoWriter object
    write_path = args['input'].split('/')[-1]
    out = cv2.VideoWriter(f"../outputs/{write_path}", 
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                          (frame_width, frame_height))

    frame_counter = 0 # to count the number of frames
    fps_countetr = 0 # to get the total FPS of all frames
    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:
            start = time.time()
            frame = cv2.resize(frame, (frame_width, frame_height)) # downscale to improve frame rate
            # conver to PIL RGB format before predictions
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated_frame = np.array(detect(pil_image, min_score=0.4, max_overlap=0.3, top_k=200))
            # convert to cv2 BGR format
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            end = time.time()
            
            cv2.imshow('image', annotated_frame)
            fps = 1 / (end - start)
            # print(f"{fps:.3f} FPS")
            out.write(annotated_frame)

            wait_time = max(1, fps/4)
            # print(f"Wait time: {wait_time:.3f}")
            # press `q` to exit
            if cv2.waitKey(int(wait_time)) & 0xFF == ord('q'):
                break
            frame_counter += 1
            fps_countetr += fps

        else: 
            break
    
    avg_fps = fps_countetr / frame_counter
    print(f"Average FPS: {avg_fps:.2f}")

    # release VideoCapture()
    cap.release()

    # close all frames and video windows
    cv2.destroyAllWindows()

    with open(file='../logs/log.txt', mode='a+') as f:
        f.writelines(f"\nNEW RUN({datetime.now()}):, \t {args['checkpoint']}, \t Trained Epochs: {start_epoch}, \t {args['input']}, \t {avg_fps:.2f}FPS \n")