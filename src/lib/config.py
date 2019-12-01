#!/usr/bin/env python
# coding: utf-8

# In[2]:


from easydict import EasyDict as edict


# In[3]:


__C = edict()
# Consumers can get config by:
cfg = __C


# In[5]:


#
# Training
#
__C.TRAIN = edict()

# Data augmentation
__C.TRAIN.RANDOM_CROP = True
__C.TRAIN.PAD_X = 9 #10
__C.TRAIN.PAD_Y = 9 #10
__C.TRAIN.FLIP = True

# For no random bg images, add random colors
__C.TRAIN.NO_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.RANDOM_BACKGROUND = False
__C.TRAIN.SIMPLE_BACKGROUND_RATIO = 0.5  # ratio of the simple backgrounded images


# In[ ]:




