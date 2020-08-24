# Core Analyses Project

## Installation notes

Install pytroch and fastai from conda:

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c fastai -c pytorch fastai
```

Other dependencies via pip

```
pip install -r requirements.txt
```

The notebook includes an animated progressbar for which we need `ipywidgets` enabled

```
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## BGS Core Processing workflow

First of all why do this? Essentially classical edge detection workflows do not work very well on images that are not pre-processed or taken in a way where there is good contrast. Picking the right threshold value is also tedious and can take some time, even contouring and finding the correct contours needs to be done per image for optimal results. 
![Example of Canny Edge detection](Images/S00128804.Cropped_Top_2.gif)
![Example of Canny Edge detection](Images/S00128815.Cropped_Top_2.gif)

Project updated to include the machine learning workflow for segmentation as well as the various other workflows that follow.

There are several files some of which are used to generate the masks that are then used to train the model and the rest are used to also get the areas from the core after its processed.

A total of 33 training images were used- 4 of which were used for validation throughout the training process - the model was then tested on additional pictures and the accuracy of the model was assesed qualitatively, depending on the accuracy for the purpose.
![Example of training image](Images/train/S00128907.Cropped_Top_1.png)

The images are then put into the machine learning model and predictions are then made based on what it has previously learnt, example of the ML output (Through segmentation and Unet as implemented in fastai)

The example below is of a prediction on an untrained example:

![Example of image processed through unet](Images/S00128821.Cropped_Top_2_resized.png)

![Example of image processed through unet](Images/OutputFromML.png)

The final output from the images themselves looks like what is below- with varying accuracies depending on how well the mask worked
![Example of fully processed image](Images/S00128821.Cropped_Top_2_Countoured.png)

***

***Dependencies***

OpenCV Above 3.4 - This is for the computer vision parts - segmentation and masking etc.. 
Sci-kit image to make the watershed segmentation work, along with imutils

FastAi - which is built on pytorch - setup uses Salamander.Ai - most of the needed libraries come pre-installed - this is to recreate the machine learning parts - the code should run assuming you have all the needed training files in the right directories

Numpy is needed for everything
