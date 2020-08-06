# SSD300-VGG11-on-Pascal-VOC-2005-Data



***This project trains a SSD300 with VGG11 base on the [PASCAL VOC 2005](http://host.robots.ox.ac.uk/pascal/VOC/voc2005/index.html) dataset using the PyTorch deep learning framework.***



![](https://github.com/sovit-123/SSD300-VGG11-on-Pascal-VOC-2005-Data/blob/master/outputs/fire_fighter.jpg?raw=true)



## <u>Table of Contents</u>

* **[Introduction](#Introduction).**

* **[Project Directory Structure.](#Project Directory Structure)**
* **[PyTorch Version.](#PyTorch Version)**
* **[How to Run?](#How to Run?)**
* **[Get Access to Pretrained Model.](#Get Access to Pretrained Model)**
* **[Results](#Results).**



## <u>Introduction</u>

***This project trains a SSD300 with VGG11 base on the [PASCAL VOC 2005](http://host.robots.ox.ac.uk/pascal/VOC/voc2005/index.html) dataset using the PyTorch deep learning framework.*** 

The VOC 2005 dataset mainly contains 4 classes for detection purposes. They are:

* Bicycle.
* Car.
* Motorbike.
* Person.



## <u>Project Directory Structure</u>

Follow the below given structure while cloning this repo or setting up the project and you should be good to go,

```
├───VOCdevkit
└───voc_2005
    ├───voc2005_1
    │   ├───Annotations
    │   ├───GTMasks
    │   └───PNGImages
    └───voc2005_2
        ├───Annotations
        ├───GTMasks
        └───PNGImages
├───logs
├───model_checkpoints
├───outputs
└───src
    │   create_data.py
    │   datasets.py
    │   detect.py
    │   detect_vid.py
    │   eval.py
    │   model.py
    │   my_utils.py
    │   train.py
    │   utils.py
```

***If you follow the above structure, then you can just clone and run the files according to the execution commands.***



## <u>PyTorch Version</u>

* PyTorch >= 1.4



## ****<u>How to Run?</u>

***Execute the following commands in the given order.***

*Note: For testing purposes, put all the image and video files inside `input/test_data` that you want to test or detect on*.

* **Run the following command only once. This prepare the train and test JSON files inside the `input` folder.**
  * `python create_data.py --input1 ../input/VOCdevkit/voc_2005/voc2005_1 --input2 ../input/VOCdevkit/voc_2005/voc2005_2 --save-path ../input`
* **To train the SSD model on the dataset**.
  * `python train.py --input ../input --continue-training yes`
  * If you are training for the first time, then give the `--continue-training` argument as `no`. This will train the model from scratch.
  * If you have trained the model for some epochs and want to continue training, then give the `--continue-training` argument as `yes`.
  * Check the number of iterations to train for inside the `train.py` file. 

* **To run the model and detection on videos.**
  * `python detect_vid.py --input ../input/test_data/video1.mp4 --resize yes`
  * In the above command, if `--resize` is `yes`, then the code will resize (make them smaller) the video frames only if the original frames' width is more than 800px.
* **To detect objects in an image**.
  * `python detect.py --input ../input/test_data/fire_fighter.jpg`
* **To run evaluation on validation data**.
  * `python eval.py`



## <u>Get Access to Pretrained Model</u>

***If you have hardware issues and cannot train the model on your system, then [here](https://drive.google.com/file/d/1Vyv7utfms98uPWpW6tfLAWVGWU2Y7E1K/view?usp=sharing) is the link to a model trained for 156 epochs.***

* [**Google Drive Link.**](https://drive.google.com/file/d/1Vyv7utfms98uPWpW6tfLAWVGWU2Y7E1K/view?usp=sharing)
* Now you can easily run the `detect_vid.py` and `detect.py` files.



## <u>Results</u>

![](https://github.com/sovit-123/SSD300-VGG11-on-Pascal-VOC-2005-Data/blob/master/outputs/fire_fighter.jpg?raw=true)



![](https://github.com/sovit-123/SSD300-VGG11-on-Pascal-VOC-2005-Data/blob/master/outputs/motor_bike1.jpg?raw=true)

![](https://github.com/sovit-123/SSD300-VGG11-on-Pascal-VOC-2005-Data/blob/master/outputs/motor_bike3.png?raw=true)

![](https://github.com/sovit-123/SSD300-VGG11-on-Pascal-VOC-2005-Data/blob/master/outputs/cars1.jpg?raw=true)