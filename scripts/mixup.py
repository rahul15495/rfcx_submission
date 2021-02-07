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

    def __call__(self, y1, y2, alpha=None):
        sr= self.sr
        if not alpha :
            alpha= np.random.uniform(0.3, 0.7)
        
        option = random.choice([0,1,2,3])
        
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
