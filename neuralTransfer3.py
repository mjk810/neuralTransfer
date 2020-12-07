#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:16:42 2020

@author: marla
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
from PIL import Image
import PIL

#code adapted from tensorflow https://www.tensorflow.org/tutorials/generative/style_transfer

class ImageTransfer:
    def __init__(self, contentImagePath, styleImagePath, content_layers, style_layers, style_weight, content_weight):
        self.contentImagePath = contentImagePath
        self.styleImagePath = styleImagePath
        self.contentImage=self.readAndFormatImages(contentImagePath)
        self.styleImage=self.readAndFormatImages(styleImagePath)
        self.style_layers=style_layers
        self.content_layers=content_layers
        self.num_style_layers=len(style_layers)
        self.num_content_layers=len(content_layers)
        self.transferModel=self.loadTransferModel()
        self.miniModel=self.setUpMiniModel()
        self.opt=self.setOptimization()
        self.style_weight=style_weight
        self.content_weight=content_weight
        self.targetImage=None
        self.targetStyle=None
        self.targetContent=None
        
        
    def createTrainingImage(self):
        #set the content image as the starting point for the new image
        self.targetImage = tf.Variable(self.contentImage)

        
    def getlayers(self):
        #utility function to print the names of the content layers 
       print("selfcontent", self.content_layers)
       print(self.num_content_layers)
             
    def printModelLayers(self):
        #utility function to print the names of the layers in the model
        for layer in self.transferModel.layers:
            print(layer.name)
   
    def displayImage(self):
        #utility function to display the original content and style images
        '''original unformatted images'''
        im=plt.imread(self.contentImagePath)
        plt.imshow(im)
        plt.show()
        plt.close()
        
        im=plt.imread(self.styleImagePath)
        plt.imshow(im)
        plt.show()
        plt.close()

        
    def readAndFormatImages(self, filePath):
        #read the images, set the datatype, and resize to 400 x 400
        image = plt.imread(filePath)
        img = tf.image.convert_image_dtype(image, tf.float32)
        img = tf.image.resize(img, [400, 400])
        # Shape -> (batch_size, h, w, d)
        img = tf.expand_dims(img, 0)
       
        return img
    
    def loadTransferModel(self):
        #load the model and set the layers to be untrainable
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg19.trainable = False
       
        return vgg19
     
    def setUpMiniModel(self):
        #create a model that takes the transfer model as input and outputs the feature maps from
        #the content and style layers
        layerNames = self.content_layers + self.style_layers
        outputs = []
        for l in layerNames:
            outputs.append(self.transferModel.get_layer(l).output)
        
        model = tf.keras.Model([self.transferModel.input], outputs)
        return model  
    
    def gram_matrix(self, input_tensor):
        #code from tensorflow https://www.tensorflow.org/tutorials/generative/style_transfer
        #perform matrix multiplication of the input_tensor by the input_tensor Transpose; 4D matrix to 3D matrix
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    
        input_shape = tf.shape(input_tensor)
        #normalize to the size of the image feature map at each layer
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        
        return result/num_locations
    
    def gram(self, input_tensor):
        #attempt at alternative gram matrix calculation based on the definition
        #and using numpy instead of tf; not yet working
        x=input_tensor.numpy()
        x = np.squeeze(x)
        ch_size = x.shape[0]*x.shape[1]
        
        x=x.reshape(x.shape[2],ch_size)
        result = x.dot(x.T)
        result=np.expand_dims(result, axis = 0)
        
        return tf.convert_to_tensor(result)
    
    
    def setOptimization(self):
        #define the parameters for the optimization algorithm
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        return opt
    
    def calculateContentLoss(self, outputs, lossType="L2"):
        #loss type parameter is a string L1 or L2 for loss
        #content loss is the squared-error loss 
        content_outputs = outputs['content']
        if lossType == "L1":
            #L1 loss is the mean absolute error
            content_loss = tf.add_n([tf.reduce_mean(tf.abs(content_outputs[name]-self.targetContent[name])) for name in content_outputs.keys()])
        else: 
            #L2 loss is the sum squared error
            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.targetContent[name])**2) for name in content_outputs.keys()])
        #normalize weight to number of layers     
        content_loss = content_loss / self.num_content_layers
        return content_loss
   
    def calculateStyleLoss(self, outputs):
        #style loss is mean square error between the gram matrix for each style layer
        #and the gram matrix for the corresponding target layer
        style_outputs = outputs['style']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.targetStyle[name])**2) 
                               for name in style_outputs.keys()])
        #normalize to the number of style layers
        style_loss = style_loss / self.num_style_layers
        return style_loss
       
    
    def computeLoss(self, outputs):
        #parameters: outputs - a dictionary of style and content outputs
        #total loss = style_weight * style loss + content_weight * content loss
        style_loss = self.calculateStyleLoss(outputs)
        #print("style_loss ", style_loss)
        content_loss = self.calculateContentLoss(outputs, "L1")
        #print("content_loss ", content_loss)
        loss = self.style_weight * style_loss + self.content_weight * content_loss
        print("loss: ", loss)
        return loss
    
    def setTrainingTargets(self):
        '''run one forward pass through the mini model and get the content and style outputs
        these will be the target values that we optimize to'''
        #the target style is the gram matrix for the style layers of the style image (used for style loss function)
        #this is used in the loss function
        self.targetContent = self.forwardPass(self.contentImage)['content']
        self.targetStyle = self.forwardPass(self.styleImage)['style']
    
    def forwardPass(self, imgTarget):
        #Expects float input in [0,1]
        #the forward pass will get the style and content feature maps from the specified
        #style and content layers of the VGG 19 model and calculate the gram matrix for the 
        #style layers
        
        imgTarget = imgTarget*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(imgTarget)
        
        #get the feature maps for style and content layers
        outputs = self.miniModel(preprocessed_input)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]
        
        #calculate the gram matrix for each feature map in the style layers
        style_outputs = [self.gram_matrix(style_output)
                         for style_output in style_outputs]
        #print("style output gram ", len(style_outputs))
        #print(style_outputs)
    
        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
    
        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        
        return {'content':content_dict, 'style':style_dict}
    
    def clip_0_1(self,image):
        #clip image values to between 0 and 1; code from documentation
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def train_step(self):
        #update the target image using gradient descent
      with tf.GradientTape() as tape:
        outputs = self.forwardPass(self.targetImage)
       
        #print("outputs ",outputs)
        loss = self.computeLoss(outputs)
    
        grad = tape.gradient(loss, self.targetImage)
        self.opt.apply_gradients([(grad, self.targetImage)])
        self.targetImage.assign(self.clip_0_1(self.targetImage))
    
    
    def train(self):

       epochs = 10
       steps_per_epoch = 100
       step = 0
       for n in range(epochs):
         for m in range(steps_per_epoch):
           step += 1
           self.train_step()
           print("epoch: ", n, " step ", m , end='')
         
         plt.imshow(self.targetImage.numpy().squeeze())
         plt.show()
         plt.close()

         print("Train step: {}".format(step))
         
    def createBlankImage(self):
        #use noise as the starting point for the new image; use this or set the target to the content image
       
        imarray=np.random.rand(400,400,3)
        imarray=imarray.astype('float32')
        imarray=np.expand_dims(imarray, axis=0)
        imarray=tf.convert_to_tensor(imarray, dtype=tf.float32)
        
        self.targetImage = tf.Variable(imarray)
        
        
    
        
        
#define path to images
style_image_path = "/home/marla/Documents/Pictures/dali.jpg"
content_image_path = "/home/marla/Documents/Pictures/louvre.jpg"

#set up layers to use for content and style - choose and change layers to see impact
#on output image
content_layers = ['block1_conv2','block2_conv2','block3_conv2']
style_layers=['block3_conv1',
              'block3_conv2',
              'block4_conv1',
              'block4_conv2',
              'block5_conv1']

#set up the weights for the loss function; change weights to change output
style_weight=1e2
content_weight=1e3

#create imagetransfer object
it=ImageTransfer(content_image_path, style_image_path, content_layers, style_layers, style_weight, content_weight)
#it.getlayers()
#it.printModelLayers()
#set the targets for style and content targets
it.setTrainingTargets()

#create the image that will be optimized
it.createTrainingImage()
#it.createBlankImage()

it.train()

