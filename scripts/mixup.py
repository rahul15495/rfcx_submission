import numpy as np
import random

class MixUp:
    def __init__(self ,denoise,sr):
        '''
        Args:
            denoise : function object to denoise input signals
        '''
        self.denoise = denoise
        self.sr= sr
        
    def pad(self, y1, y2):
        max_shape= max(y1.shape[0], y2.shape[0])

        a= np.zeros(max_shape)
        b= np.zeros(max_shape)

        a[:y1.shape[0]] = y1
        b[:y2.shape[0]] = y2
        return a,b

    def __call__(self, y1, y2, alpha=None):
        sr= self.sr
        if not alpha :
            alpha= np.random.uniform(0.3, 0.7)
        
        option = random.choice([0,1,2,3])
        
        if y1.shape!=y2.shape:
            a,b= self.pad(y1, y2)
        else:
            a,b= y1.copy() , y2.copy()
        
        if option==1:
            a= self.denoise(a, sr)
        elif option==2:
            b= self.denoise(b, sr)
        elif option==3:
            a= self.denoise(a, sr)
            b= self.denoise(b, sr)
        else:
            #option==0
            pass
        
        y= alpha*a +(1-alpha)*b
        return alpha,y
