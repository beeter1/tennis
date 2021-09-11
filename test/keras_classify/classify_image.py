import tensorflow  as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import argparse
import cv2
import os
from matplotlib import pyplot as plt

# argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
# arg_img="/home/fd/code/car.png"
# arg_img = "/home/fd/code/test_images/15.jpg"

# ap.add_argument("-m", "--model", type=str, default="resnet", help="name of pre-trained network to use")
ap.add_argument("-m", "--model", type=str, default="xception", help="name of pre-trained network to use")
# args = vars(ap.parse_args())
args =  vars(ap.parse_known_args()[0])

# defind a dict that maps model name to their classes inside Keras
MODELS= {
    "resnet": ResNet50,
    "inception": InceptionV3,
    "xception": Xception,
    "vgg16": VGG16,
    "vgg19": VGG19
}

# checking  validation of model name
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in MODEL")
    
# initialize input image shape according to which model we choose
if args["model"] in ("inception", "xception"):
    input_shape = (299, 299)
    preprocess = preprocess_input
else:
    input_shape = (224, 224)
    preprocess = imagenet_utils.preprocess_input
    
# load weight  (it will take a period of time to dlownload weights if it is the first time to run this script)
print("============[INFO] loading {}...".format(args["model"]))
net = MODELS[args["model"]]
model = net(weights="imagenet")

# load the input image with the help of keras.preprocessing image
print("============[INFO] loading and preprocessing image...")
# image = load_img(args["image"], target_size=input_shape)
image = load_img(args["image"], target_size=input_shape)
image = img_to_array(image)

# add a dimension as the batch dim
image = np.expand_dims(image, axis=0)
# image = tf.concat([image, image, image, image, image, image], axis=0)

# print(tf.shape(image))

# preprocess the image
image = preprocess(image)

# classify the image
print("============[INFO] classifying  image using '{}'...".format(args["model"]))
preds = model.predict(image)
pred_info = imagenet_utils.decode_predictions(preds)

# output rank-5 predictions + probabilites to terminal
for (i, (imagenetID, label, prob)) in enumerate(pred_info[0]):
    print("{}, {}: {:.2f}%".format(i+1, label, prob*100))
    
# load image using OpenCV.
# draw the top prediction on the image
# display the image on the screen
cv_img = cv2.imread(args["image"])
(imageID, label, prob) = pred_info[0][0]
# cv2.putText(cv_img, "Label: {}, {:.2f}%".format(label, prob*100)， (10, 30), cv2.FONT_NERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
# cv2.imshow("Classification Result", cv_img)
# cv2.waitKey(0)

plt.imshow(cv_img)
print("============\033[33;1m [RESULT] classification reslut: {} \033[0m".format(label))


