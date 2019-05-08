import torch
from torch import nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_with_regressor(nn.Module):
    """
    Encoder.
    """

    def __init__(self,input_size = 162):
        super(CNN_with_regressor, self).__init__()
                
        squeezenet = models.squeezenet1_0(pretrained=False)

        
        # Remove linear and pool layers (since we're not doing classification)
        # Add AdaptiveAvgPool2d for arbitary size of input
        modules = list(squeezenet.children())[:-1] \
                  +[nn.AdaptiveAvgPool2d((1, 1))]
                  
        self.squeezenet = nn.Sequential(*modules)        
        
       
        
        # Resize image to fixed size to allow input images of variable size
        self.regressor = nn.Sequential(nn.Linear(512,512),
                                       nn.Tanh(),
                                       nn.Linear(512,256),
                                       nn.Tanh(),
                                       nn.Linear(256,1),  
                                       nn.Tanh()
                                       )
    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 1, image_size, image_size)
        :return: pred_corrs
        """
        
        out = self.squeezenet(images)  # (batch_size, 512, image_size/32, image_size/32)

        out = out.view(out.size(0),-1)        

        out = self.regressor(out)      # (batch_size, 1)        

        return out


        