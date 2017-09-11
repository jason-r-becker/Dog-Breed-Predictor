import matplotlib.pyplot as plt
import numpy as np
import pylab
import random
import cv2
import warnings

from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from extract_bottleneck_features import extract_Resnet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from glob import glob
from tqdm import tqdm
from PIL import Image

# Settings
# ------------------------------------------------------------------------------------------------------------------ #
filenames = ['megan2', 'megan dog']
save = True
# ------------------------------------------------------------------------------------------------------------------ #


warnings.filterwarnings("ignore")
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def cnn_model():
    model = Sequential([
        GlobalAveragePooling2D(input_shape=train_resnet.shape[1:]),

        BatchNormalization(),
        Dropout(0.2),
        Dense(256, activation='relu'),

        BatchNormalization(),
        Dropout(0.2),
        Dense(133, activation='softmax')
        ])
    return model


def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
bottleneck_features = np.load('DogResnet50Data.npz')
train_resnet = bottleneck_features['train']
resnet_model = cnn_model()
resnet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
resnet_model.load_weights('saved models/weights.best.Resnet50.hdf5')


# Extract bottleneck features
def resnet_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = resnet_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


# make prediction
def predict_image(img_path):
    if dog_detector(img_path):
        pred = resnet_predict_breed(img_path)
        print("This dog looks like a {}.".format(pred.replace('_', ' ')))

        f = pylab.figure()
        input_img = Image.open(img_path)
        f.add_subplot(1, 2, 1)
        pylab.imshow(np.asarray(input_img))

        pred_img_path = random.choice(glob('dogImages/train/???.{}/*'.format(pred)))
        pred_img = Image.open(pred_img_path)
        f.add_subplot(1, 2, 2)
        pylab.imshow(np.asarray(pred_img))
        plt.suptitle('This dog looks like a {}'.format(pred.replace('_', ' ')))
        return 'dog'

    elif face_detector(img_path):
        pred = resnet_predict_breed(img_path)
        print("This human most resembles a {}.".format(pred.replace('_', ' ')))

        f = pylab.figure()
        input_img = Image.open(img_path)
        f.add_subplot(1, 2, 1)
        pylab.imshow(np.asarray(input_img))

        pred_img_path = random.choice(glob('dogImages/train/???.{}/*'.format(pred)))
        pred_img = Image.open(pred_img_path)
        f.add_subplot(1, 2, 2)
        pylab.imshow(np.asarray(pred_img))
        plt.suptitle('This human most resembles a {}'.format(pred.replace('_', ' ')))

        return 'human'
    else:
        print("Error: Please input a picture of either a dog or human face.")
        return 'error'
for file in filenames:
    result = predict_image('images/{}.png'.format(file))
    if result != 'error':
        plt.tight_layout()
        if save:
            plt.savefig('saved results/{}/{}.png'.format(result, file), bbox_inches='tight', format='png')
        plt.show()
