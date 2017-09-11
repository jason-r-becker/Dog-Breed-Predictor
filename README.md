# Dog-Breed-Predictor
Convolutional Neural Network (CNN) which classifies input images as either a dog or human, and then determines the breed of the dog for dog images or the breed of dog which most resembles the person for human images.

Dog detection is performed using a ResNet with 50 layers, while human face detection is performed using OpenCV's cascade classifier. The primary model is a transformation network combining a 50 layer ResNet with 1 additional layer before a softmax output layer. The model achieved a test accuracy on dogs of 81%.
