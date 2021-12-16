# pytorch-implementations
Epic PyTorch Implementations of common ML Architectures
 - ResNets
These consist of residual blocks that allow for deeper networks (soetimes with 100+ layers) to converge well. 
 - AlexNet
 This network is one of the classic and seminal works in Convolutional Neural Networks.
 - VGG16
 Another one of the classic CNNs.
 - GoogLeNet
 Introduced Network in Network (NiN) modules for CNNs. Uses two auxiliary branches for predictions during training to combat gradient loss. During evaluation, only one branch is used.

# Upcoming
GoogLeNet
- Resolve the issue around how to use certain branches for training only.
- https://stackoverflow.com/questions/67789463/how-to-discard-a-branch-after-training-a-pytorch-model

Faster R-CNN
- Need to Implement:
- Feature Pyramid Network (hopefully actual architecture will be easy to code )
- Define, use anchor boxes (This part is what I anticipate to be challenging. For some reason, anchor boxes are not really intuitive to me.) I may need to write a kind of description after reading some papers to cement my understanding.

Fully Convolutional Network
- This, I feel will be important for the monocular depth perception project that I'm doing.
- Will use this for the semantic segmentation part of the model


