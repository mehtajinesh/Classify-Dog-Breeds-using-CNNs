import numpy as np
from glob import glob #read images into numpy
import cv2                
import os
import matplotlib.pyplot as plt
from torch import cuda
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import torch
from PIL import Image, ImageFile

from CustomCNN import Net
import torch.optim as optim
import torch.nn as nn


ImageFile.LOAD_TRUNCATED_IMAGES = True                        
'exec(%matplotlib inline)'
DEBUG = False
cwd = os.getcwd()
use_cuda = cuda.is_available()

def write_log(msg):
    ''' Write logs for debugging
    '''
    print(msg)
    
def read_files():
    ''' Reads the images from the local device to the
        application.
    '''
    # load filenames for human and dog images
    humans = np.array(glob(cwd+"\\lfw\\*\\*"))
    dogs = np.array(glob(cwd+"\\dogImages\\*\\*\\*"))
    if DEBUG:
        # print number of images in each dataset
        write_log('There are %d total human images.' % len(humans))
        write_log('There are %d total dog images.' % len(dogs))
    return humans,dogs

def test_pretrained_face_detector(human_files, dog_files, bShowSample=False):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cwd+ "\\haarcascades\\haarcascade_frontalface_alt.xml")
    # load color (BGR) image
    img = cv2.imread(human_files[0])
    '''Before using any of the face detectors, it is standard procedure to convert the images to grayscale. The detectMultiScale function executes the classifier stored in face_cascade and takes the grayscale image as a parameter.'''
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find faces in image
    faces = face_cascade.detectMultiScale(gray)
    if DEBUG:
        # print number of faces detected in the image
        write_log('Number of faces detected:', len(faces))
    # get bounding box for each detected face
    for (x,y,w,h) in faces:
        # add bounding box to color image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    if bShowSample:
        # convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # display the image, along with bounding box
        plt.imshow(cv_rgb)
        plt.show()

def face_detector(img_path):
    '''
    returns "True" if face is detected in image stored at img_path
    '''
     # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cwd+'\\haarcascades\\haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def calc_accuracy_face_detector(human_files_short, dog_files_short):
    ''' Test the performance of the face_detector algorithm 
        on the images in human_files_short and dog_files_short.
    '''
    print('Calculating Accuracy for Haar Face Detection using {}'.format(len(human_files_short)))
    human_detected = 0.0
    dog_detected = 0.0
    num_files = len(human_files_short)
    for i in range(0, num_files):
        human_path = human_files_short[i]
        dog_path = dog_files_short[i]
        if face_detector(human_path) == True:
            human_detected += 1
        if face_detector(dog_path) == True:
            dog_detected += 1
    print('Haar Face Detection')
    print('The percentage of the detected face - Humans:{0:.0%}'.format(human_detected / num_files))
    print('The percentage of the detected face - Dogs:{0:.0%}'.format(dog_detected / num_files))

def VGG16_predict(img_path, VGG16):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    '''
    ## Return the *index* of the predicted class for that image
    img = Image.open(img_path)
    min_img_size = 224
    transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    if use_cuda:
        img = img.to('cuda')
    prediction = VGG16(img)
    prediction = prediction.argmax()
    return prediction # predicted class index

def dog_detector(img_path):
    '''Given an image, this function returns "True" if a dog is detected in the image stored at img_path (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image using pre-trained VGG-16 model. 
    '''
    # define VGG16 model
    VGG16 = models.vgg16(pretrained=True)
    # move model to GPU if CUDA is available
    if use_cuda:
        VGG16 = VGG16.cuda()
    prediction_index = VGG16_predict(img_path,VGG16).data.cpu().numpy()
    is_dog = 151 <= prediction_index <= 268
    return is_dog  # true/false

def data_loader_dog_dataset():
    ''' Creates three separate data loaders for the training, validation, and test datasets of dog images (located at dog_images/train, dog_images/valid, and dog_images/test, respectively). Also, augmenting your training and/or validation data using transforms
    '''
    data_dir = cwd+'/dogImages'
    TRAIN = 'train'
    VAL = 'valid'
    TEST = 'test'

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    input_shape = 224
    batch_size = 32
    scale = 256

    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        VAL: transforms.Compose([
            transforms.Resize(scale),
            transforms.CenterCrop(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        TEST: transforms.Compose([
            transforms.Resize(scale),
            transforms.CenterCrop(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, VAL, TEST]
    }

    loaders_scratch = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=0)
        for x in [TRAIN, VAL, TEST]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

    for x in [TRAIN, VAL, TEST]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))
        
    print("Classes: ")
    class_names = image_datasets[TRAIN].classes
    print(image_datasets[TRAIN].classes)
    return loaders_scratch

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """ returns trained model
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            val_outputs = model(data)
            val_loss = criterion(val_outputs, target)
            valid_loss += ((1 / (batch_idx + 1)) * (val_loss.data - valid_loss))

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## save the model if validation loss has decreased
        if(valid_loss < valid_loss_min):
            print('Saving model: Validation Loss: {:.6f} decreased \tOld Validation Loss: {:.6f}'.format(valid_loss, valid_loss_min))
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
            
    # return trained model
    return model

if __name__ == "__main__":
    '''
    run this module if the call comes from main
    (avoids getting called when used for importing)
    '''
    print("Face Detection using pretrained model- haar cascade classifier")
    if DEBUG:
        write_log("Importing datasets")
    human_files, dog_files = read_files()
    if DEBUG:
        write_log("Imported datasets")

    ''' First we detect human faces using pre-trained face detectors
        provided by OpenCV's Haar feature-based cascade classifier.    
    '''
    #test_pretrained_face_detector(human_files,dog_files)
    ''' Calculate the accuracy of the pre-trained model on 100 samples
        from dog and human images
    '''
    print("Testing...")
    human_samples = human_files[:100]
    dog_samples = dog_files[:100]
    #calc_accuracy_face_detector(human_samples,dog_samples)
    ''' Now we detect dog images using VGG-16 model along with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories..
    Lets test the performance of the dog_detector function on the images in human_files_short and dog_files_short
    '''
    print("Dog Detection using pretrained model- VGG16")
    print("Testing...")
    detected_humans = []
    detected_dogs = []
    for human, dog in zip(human_samples, dog_samples):
        #detected_humans.append(dog_detector(human))
        #detected_dogs.append(dog_detector(dog))
        pass
    print(f'Humans detected as dogs: {np.sum(detected_humans)/len(detected_humans):.2%}')
    print(f'Dogs detected correctly: {np.sum(detected_dogs)/len(detected_dogs):.2%}')
    
    '''
    The architecture is composed from a feature extractor and a classifier. The feature extractor is has 3 CNN layers in to extract features. Each CNN layer has a ReLU activation and a 2D max pooling layer in to reduce the amount of parameters and computation in the network. After the CNN layers we have a dropout layer with a probability of 0.5 in to prevent overfitting and an average pooling layer to calculate the average for each patch of the feature map. The classifier is a fully connected layer with an input shape of 64 x 14 x 14 (which matches the output from the average pooling layer) and 133 nodes, one for each class (there are 133 dog breeds). We add a softmax activation to get the probabilities for each class.
    '''
    # data for training, validating and test custom cnn arch
    loaders_scratch = data_loader_dog_dataset()
    print("Data loaded for training custom cnn") 
    # instantiate the Custpm CNN Arch.
    model_scratch = Net()
    # move tensors to GPU if CUDA is available
    if use_cuda:
        model_scratch.cuda()
    # select loss function
    criterion_scratch = nn.CrossEntropyLoss()
    print("Cross Entropy Loss Loaded.") 
    # select optimizer
    optimizer_scratch = optim.Adam(model_scratch.parameters())
    print("Adam Optimizer selected.") 
    # train the model
    print("Training started...") 
    model_scratch = train(50, loaders_scratch, model_scratch, optimizer_scratch, criterion_scratch, use_cuda, 'model_scratch.pt')
    print("Training completed.") 
    # load the model that got the best validation accuracy
    model_scratch.load_state_dict(torch.load('model_scratch.pt'))
    print("Best model loaded.") 
    




