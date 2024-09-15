#import sys
#sys.path.append('~/xiaoni/MCD_DA-master/classification/model')
from usps import*
import usps
#from .usps import AdversarialNetwork 
#from .usps import Predictory 
#from .usps import Mine_1


import svhn2mnist
#from .usps import AdversarialNetwork 

import model.syn2gtrsb as syn2gtrsb
#from .syn2gtrsb import*
#from .syndig2svhn import*

def discriminator_mi(source, target):
    if source == 'USPS' or target == 'USPS':
        return usps.Domain_discriminator(48*4*4)
    if source == 'SVNH':
        return svhn2mnist.Domain_discriminator(3072)
    if source == 'SYNTH':
        return syn2gtrsb.Domain_discriminator( )


def discriminate_Type(source, target, pixelda=False):
    if source == 'USPS' or target == 'USPS':
        return usps.discriminate_Type_mi()
    elif source == 'SVNH':
        return svhn2mnist.discriminate_Type_mi()
    elif source == 'SYNTH':
        return syn2gtrsb.discriminate_Type_mi()

def Generator(source, target, pixelda=False):
    if source == 'USPS' or target == 'USPS':
        return usps.Feature()
    elif source == 'SVNH':
        return svhn2mnist.Feature()
    elif source == 'SYNTH':
        return syn2gtrsb.Feature()


def Classifier(source, target):
    if source == 'USPS' or target == 'USPS':
        return usps.Predictor()
    if source == 'SVNH':
        return svhn2mnist.Predictor()
    if source == 'SYNTH':
        return syn2gtrsb.Predictor()

def Classifier_y(source, target):
    if source == 'USPS' or target == 'USPS':
        return usps.Predictory()
    if source == 'SVNH':
        return svhn2mnist.Predictor()
    if source == 'SYNTH':
        return syn2gtrsb.Predictor()

def discriminator(source, target):
    if source == 'USPS' or target == 'USPS':
        return usps.AdversarialNetwork(48*4*4)
    if source == 'SVNH':
        return svhn2mnist.AdversarialNetwork(3072)
    if source == 'SYNTH':
        return syn2gtrsb.AdversarialNetwork( )




