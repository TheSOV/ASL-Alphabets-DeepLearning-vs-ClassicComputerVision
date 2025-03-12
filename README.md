# Real-time-detection-of-ASL-Alphabets
### Project Description:

Make a comparation of the performance of Classical Computer Vision vs Deep Learning in signs recognition.
### Objective:

In this project, a deep learning model is built to alleviate the above problem a bit. Deep learning and convolutional neural networks is used to build a model that can recognize the American Sign Language alphabets with decent enough accuracy.

### Dataset:

**Link to the dataset** - https://www.kaggle.com/grassknoted/asl-alphabet

The dataset contains 87000 images with a dimension of 200×200.
There are 29 classes in total. 26 of these classes are letters from A-Z. Then there are three more classes that correspond to  SPACE, DELETE, and NOTHING. There are 3000 images from each class.

Since the dataset is very large, it will take much more time and resources to train the model. Hence only a subset is used.

### Tools used:
- Pytorch : Creates customised CNN model 
- OpenCV : For processing images and real-time capture



