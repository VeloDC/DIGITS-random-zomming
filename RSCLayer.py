import caffe
import random
import numpy as np
import math
import cv2


class RSCLayer(caffe.Layer):

    def setup(self,bottom,top):
        assert len(bottom) == 1,            'requires a single layer.bottom'
        assert bottom[0].data.ndim >= 3,    'requires image data'
        assert len(top) == 1,               'requires a single layer.top'
        # Define crop size boundaries
        self.img_size = bottom[0].data.shape[2]
        self.crop_max_size = self.img_size
        self.crop_min_size = int(self.img_size * 1 / 2)
        # Define output image dimensions
        self.output_dim = (227,227)
        top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1],self.output_dim[0],self.output_dim[1])

    def reshape(self,bottom,top):
        pass

    def forward(self,bottom,top):
        #Random size
        crop_size = np.random.randint(self.crop_min_size, self.crop_max_size)
        #Random position
        x = np.random.randint(0, self.img_size - crop_size)
        y = np.random.randint(0, self.img_size - crop_size)
        #Cropping
        cropped_blob = bottom[0].data[:,:,x:x+crop_size,y:y+crop_size]
        #Resizing to expected dimensions
        for i in range(bottom[0].data.shape[0]):
            top[0].data[i,:,:,:] = cv2.resize(cropped_blob[i].transpose(1,2,0),self.output_dim,interpolation=cv2.INTER_LINEAR).transpose(2,0,1)

    def backward(self, top, propagate_down, bottom):
        pass
