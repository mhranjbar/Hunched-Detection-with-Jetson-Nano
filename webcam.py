# mhranjbar, All rights reserved.
# https://github.com/mhranjbar/

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
import PIL
#from PIL import Image



def preprocess(image):

    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def webcam(webcamNum = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load("my_model.pth"))
    model.to(device)
    model.eval()

    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw ! '
               'videoconvert ! appsink').format(webcamNum)
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)



    while True:
        try:
            _, frame = cap.read()

            image = frame.copy()
            shape = image.shape

            if shape[0] != 224 or shape[1] != 224:
                image = cv2.resize(image, (224,224)) 



            preprocessed = preprocess(image)
            output = model(preprocessed)
            output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()

        

            if np.argmax(output) == 0:
                print("Hunched, ","%.2f"%output[0])
                cv2.putText(frame, "Hunched, "+"%.2f"%output[0], (10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                print("Not Hunched, ","%.2f"%output[1])
                cv2.putText(frame, "Not Hunched, "+"%.2f"%output[1], (10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            #frame = Image.fromarray(frame, "RGB")
            #frame.show()
            
            
            cv2.imshow("Live" , frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        except Exception as e:
            print(e)




webcam(0)