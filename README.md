# pytorch-implementations
Epic PyTorch Implementations of common ML Architectures
 - ResNets
These consist of residual blocks that allow for deeper networks (sometimes with 100+ layers) to converge well. 
 - AlexNet
 This network is one of the classic and seminal works in Convolutional Neural Networks.
 - VGG16
 Another one of the classic CNNs.
 - GoogLeNet
 Introduced Network in Network (NiN) modules for CNNs. Uses two auxiliary branches for predictions during training to combat gradient loss. During evaluation, only one branch is used.

# Upcoming
GoogLeNet 
- At this point, just need to implement the training file for this model.

Faster R-CNN
- Need to Implement:
- I may need to write a kind of description after reading some papers to cement my understanding 
- A description of the workings of Faster R-CNN has been written in the model's folder
- One big breakthrough in my understanding of anchor boxes is that they are predicting offsets to the anchor boxes rather than directly predicting the bounding box coordinates. 


Fully Convolutional Network
- This, I feel will be important for the monocular depth perception project that I'm doing.
- Will use this for the semantic segmentation part of the model
- It is really interesting to see that the data is shown as images with the masks already drawn.
- Using color codes, we just need to make labels of the images.
- Possible model to implement: U-Net
   
 

 
