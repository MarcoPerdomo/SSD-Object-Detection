# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable #this library converts a torch tensor into a torch variable that includes the GRADIENT and the tensor
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
def detect(frame, net, transform): # net is SSD neural network. transform makes sure the images have the right format
    height, width = frame.shape[:2] #taking the height and width from original image. Third element that is excluded is the channels
    frame_t = transform(frame)[0] #transforming the frame into the right dimensions and color values
    x = torch.from_numpy(frame_t).permute(2,0,1) #converting a numpy array into a torch tensor
    # We also convert from red blue green to GRB to be consistent with what it was trained
    x = Variable(x.unsqueeze(0)) #transforming the tensor into a torch variable
    #index 0 is the new batch, it should always be the first dimension
    #adding a new dimension with batch. UNsqueezed function adds this new dimension that will be fed into the NN
    y = net(x) #feeding x to the neural network. Will return the output y
    detections = y.data #obtaining the attribute data from y
    scale = torch.Tensor([width, height, width, height])    #normalizing the scale values of width and heights from 0 to 1. The first WxH is the top left corner, second pair is bottom right corner
    #so far Detections = [batch, number of classes that can be detected, number of occurence, tuple(score, x0, y0, x1, y1)]
    for i in range(detections.size(1)): #iterating through all the classes 
        j = 0 #occurence of the class 
        while detections[0,i,j,0] >=0.6: #Checking if the score is bigger than treshold 0.6. batch 0, class i, occurence, score. In that order
            pt = (detections[0,i,j,1:] * scale).numpy() #keeping the coordinates x0, y0, x1, y1. Applying the normalization and converting back to numpy
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0,0), 2) # drawing the rectangle
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            j += 1
    return frame 
    

# Creating the SSD neural network
net = build_ssd('test')
#Loading the weights and attributing the values of the weights to the NN net
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) 


# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) #target size of the images to be given to the NN, the next argument is a convention value


# Doing some Object Detection on a video
reader = imageio.get_reader('Kenja1.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('outputKenja1.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame) # appending the detected frame into the output vid
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video. 


