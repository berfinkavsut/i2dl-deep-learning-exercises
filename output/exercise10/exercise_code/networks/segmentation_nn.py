"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
import numpy as np 

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################

        #input dimension: (N,C,H,W)
        #output dimensions: (N,num_classes,H,W)
        depth = hparams['channels']
        height = hparams['height']
        width = hparams['width']
                
        self.image_backbone = models.alexnet(pretrained=True)
        self.image_backbone.classifier = nn.Identity()
        self.image_backbone.avgpool = nn.Identity()
        self.image_backbone.features[12] = nn.Identity()
        
        # self.image_backbone = models.alexnet(pretrained=True)
        # modules = list(self.image_backbone.children())[:-1]
        # self.image_backbone = torch.nn.Sequential(*modules)

        for p in self.image_backbone.parameters():
            p.requires_grad = False

        # print(self.image_backbone)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3,3),stride=(2, 2), padding=(1, 1) ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3,3),stride=(2, 2), padding=(1, 1) ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,3),stride=(2, 2), padding=(1, 1) ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
          
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3,3),stride=(2, 2), padding=(0, 0) ),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            
            nn.Upsample(scale_factor=(1.1, 1.1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5,5),stride=(1, 1), padding=(0, 0) ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
          
            nn.ConvTranspose2d(in_channels=64, out_channels=23, kernel_size=(5,5),stride=(1, 1), padding=(0, 0) ),
            nn.ReLU(inplace=True),
            )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.image_backbone(x)
        #print(x.shape)
        x = torch.reshape(x,(-1,256,196)).view(1,256,14,14)
        x = self.decoder(x)
        return x

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def training_step(self, batch, batch_idx):
      
      images, targets = batch   
      # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      # images, targets = images.to(device), targets.to(device)

      # Perform a forward pass on the network with inputs
      out = self.forward(images)
    
      # calculate the loss with the network predictions and ground truth targets
      loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
      loss = loss_func(out, targets)

      # Find the predicted class from probabilities of the image belonging to each of the classes
      # from the network output
      _, preds = torch.max(out, 1)

      # Calculate the accuracy of predictions
      targets_mask = targets >= 0
      acc = np.mean((preds.cpu() == targets.cpu())[targets_mask].numpy())

      # Log the accuracy and loss values to the tensorboard
      self.log('loss', loss)
      self.log('acc', acc)

      return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
      images, targets = batch   
      # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      # images, targets = images.to(device), targets.to(device)

      # Perform a forward pass on the network with inputs
      out = self.forward(images)

      # calculate the loss with the network predictions and ground truth targets
      loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
      loss = loss_func(out, targets)

      # Find the predicted class from probabilities of the image belonging to each of the classes
      # from the network output
      _, preds = torch.max(out, 1)

      # Calculate the accuracy of predictions
      targets_mask = targets >= 0
      acc = np.mean((preds.cpu() == targets.cpu())[targets_mask].numpy())

      return {'val_loss': loss, 'val_acc': acc}
    
    def training_epoch_end(self, outputs):

      # Average the loss over the entire validation data from it's mini-batches
      avg_loss = 0
      avg_acc = 0
      for x in outputs:
        avg_loss += x['loss']
        avg_acc += x['acc']
      avg_loss /=len(outputs)
      avg_acc /=len(outputs)

      #avg_loss = ([x['loss'] for x in outputs]).mean()
      #avg_acc =  ([x['acc'] for x in outputs])).mean()

      # Log the validation accuracy and loss values to the tensorboard
      self.log('loss', avg_loss)
      self.log('acc', avg_acc)
      
    def validation_epoch_end(self, outputs):

      # Average the loss over the entire validation data from it's mini-batches
      avg_loss = 0
      avg_acc = 0
      for x in outputs:
        avg_loss += x['val_loss']
        avg_acc += x['val_acc']
      avg_loss /=len(outputs)
      avg_acc /=len(outputs)
  
      #avg_loss = ([x['loss'] for x in outputs]).mean()
      #avg_acc =  ([x['acc'] for x in outputs])).mean()

      # Log the validation accuracy and loss values to the tensorboard
      self.log('val_loss', avg_loss)
      self.log('val_acc', avg_acc)
      

    def configure_optimizers(self):
      optim = torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])
      return optim
        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
