# mhranjbar, All rights reserved.
# https://github.com/mhranjbar/

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import PIL
import time



def preprocess(image):

    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]




def train(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    ##################### Dataset ####################
    data_path = 'datasetC/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize(size=(224, 224)), torchvision.transforms.ToTensor()])
    )

    print(train_dataset)
    #print("###############")
    print("classes: ",train_dataset.class_to_idx)

    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=5,
    shuffle=True)

    ##################### testSet ####################
    data_path = 'testSetC/'
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize(size=(224, 224)), torchvision.transforms.ToTensor()])
    )



    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=5,
    shuffle=True)



    ################### model  train #############################

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    #model.load_state_dict(torch.load("my_model.pth"))
    model.to(device)
    model = model.train()


    if torch.cuda.is_available():
        par = True
    else:
        par = False
    
    accrTemp = 0
    total_train = 0
    correct_train = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(epochs): # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            print("##########################################")
            inputs, labels = data
            
            if par:
                inputs, labels = inputs.cuda(), labels.cuda() # add this line



            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print("epoch: ",epoch+1, "/",epochs,", batch: ", i+1,", loss: ", loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()


        # accuracy
        _, predicted = torch.max(outputs.data, 1)
        #total_train += labels.size(0)
        total_train += labels.nelement()
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        #avg_accuracy = train_accuracy / len(train_loader)                                     
        print("\n")
        print('Epoch {}, train Loss: {:.3f}'.format(epoch+1, loss.item()), "Training Accuracy: %d %%" % (train_accuracy))
        accr = evalModel(model, test_loader, par, state = "test")
        print("\n")
        
        
        if accr > accrTemp and train_accuracy > 97:
            torch.save(model,str(int(accr)) + "_trainedModel.pth")
            torch.save(model.state_dict(), str(int(accr)) + "_trainedModel.pth")
            accrTemp = accr
        


    

def evalModel(model, test_loader, par, state = "test"):
    ################## model evaluation #######################
    print("Evaluating...")
    correct, total = 0, 0
    predictions = []
    model.eval()

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data

        if par:
            inputs, labels = inputs.cuda(), labels.cuda() # add this line

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    per = 100 * correct / total
    if state == "test":
        print('\n The testing set accuracy of the network is: %d %%' % (per))
    else:
        print('\n The training set accuracy of the network is: %d %%' % (per))

    
    model = model.train()
    time.sleep(0.2)
    return per


train(520)
        