from utils import label_map

import os
import json
import re
import argparse

def parse_text_files(annotation_path, obj_type):
    """
    Function to parser the dataset text files. You can find 
    more details about the datset here:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2005/chapter.pdf
    """
    boxes = list()
    labels = list()
    with open(annotation_path) as file:
        lines = file.readlines()

    for line in lines:
        if 'Bounding box' in line:
            label = obj_type 
            
            colon_split = line.split(':')[-1] # split by colon sign, :
            space_split = re.split('\s|(?<!\d)[,.]|[,.](?!\d)|\(|\)', colon_split)
            # print('LINE', space_split[2], space_split[4], space_split[8], space_split[10])
            xmin = int(space_split[2]) - 1
            ymin = int(space_split[4]) - 1
            xmax = int(space_split[8]) - 1
            ymax = int(space_split[10]) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[label])

    return {'boxes': boxes, 'labels': labels}

def create_data(voc2005_1_path, voc2005_2_path, output_folder):
    """
    This function creates lists of images, bounding boxes, and  the 
    corresponding labels and saves them as JSON files.

    :param voc07_path: path to `voc2005_1` folder
    :param voc12_path: path to `voc2005_2` folder
    :param output_folder: path to save the JSON files
    """
    voc2005_1_path = os.path.abspath(voc2005_1_path)
    voc2005_2_path = os.path.abspath(voc2005_2_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # training data
    for path in [voc2005_1_path, voc2005_2_path]:

        data_folders = os.listdir(path+'/Annotations')

        for data_folder in data_folders:
            # ignore the test images folder
            if 'test' not in data_folder.lower():
                print(data_folder)
                if 'car' in data_folder.lower():
                    print('GOT CAR')
                    obj_type = 'car'
                elif 'voiture' in data_folder.lower():
                    print('GOT CAR')
                    obj_type = 'car'
                elif 'moto' in data_folder.lower():
                    print('GOT MOTORBIKE')
                    obj_type = 'motorbike'
                elif 'person' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'pieton' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'pedestrian' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'bike' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                elif 'velo' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                elif 'bicycle' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                

                text_files = os.listdir(os.path.join(path+'/Annotations', data_folder))
                for file in text_files:
                    # parse text files
                    objects = parse_text_files(os.path.join(path+'/Annotations', 
                                                    data_folder, file.split('.')[0] + '.txt'), 
                                                    obj_type)
                    if len(objects) == 0:
                        continue
                    n_objects += len(objects)
                    train_objects.append(objects)
                    train_images.append(os.path.join(path, 'PNGImages', data_folder, file.split('.')[0] + '.png'))

    assert len(train_objects) == len(train_images)

    # save training JSON files and label map
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  

    print(f"Total training images: {len(train_images)}")
    print(f"Total training objects: {n_objects}")
    print(f"File save path: {os.path.abspath(output_folder)}")


    # test data
    test_images = list()
    test_objects = list()
    n_objects = 0

    for path in [voc2005_1_path, voc2005_2_path]:

        data_folders = os.listdir(path+'/Annotations')

        for data_folder in data_folders:
            # ignore the test images folder
            if 'test' in data_folder.lower():
                print(data_folder)
                if 'car' in data_folder.lower():
                    print('GOT CAR')
                    obj_type = 'car'
                elif 'voiture' in data_folder.lower():
                    print('GOT CAR')
                    obj_type = 'car'
                elif 'moto' in data_folder.lower():
                    print('GOT MOTORBIKE')
                    obj_type = 'motorbike'
                elif 'person' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'pieton' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'pedestrian' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'bike' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                elif 'velo' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                elif 'bicycle' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                

                text_files = os.listdir(os.path.join(path+'/Annotations', data_folder))
                for file in text_files:
                    # parse text files
                    objects = parse_text_files(os.path.join(path+'/Annotations', 
                                                    data_folder, file.split('.')[0] + '.txt'), 
                                                    obj_type)
                    if len(objects) == 0:
                        continue
                    n_objects += len(objects)
                    test_objects.append(objects)
                    test_images.append(os.path.join(path, 'PNGImages', data_folder, file.split('.')[0] + '.png'))

    assert len(test_objects) == len(test_images)

    # save test JSON files
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print(f"Total test images: {len(test_images)}")
    print(f"Total test objects: {n_objects}")
    print(f"File save path: {os.path.abspath(output_folder)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input1', help='path to VOC 2005_1 dataset')
    parser.add_argument('-i2', '--input2', help='path to VOC 2005_2 dataset')
    parser.add_argument('-s', '--save-path', dest='save_path', 
                        help='path to save the created JSON files')
    args = vars(parser.parse_args())

    create_data(
        voc2005_1_path=args['input1'],
        voc2005_2_path=args['input2'],
        output_folder=args['save_path']
    )