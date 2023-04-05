# -*- coding: utf-8 -*-
"""Math 373.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IfcfCpdDE7gny9Mpj5NHlQncvoP5zOIs
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import sklearn.model_selection
import torch

df_labeled = pd.read_csv('./sample_data/mnist_train_small.csv', header=None)
df_test = pd.read_csv('./sample_data/mnist_test.csv', header=None)

df_train, df_val = sk.model_selection.train_test_split(df_labeled)

df_val.shape

#We need to define a dataset class with three methods
class DigitsDataset(torch.utils.data.Dataset):

  def __init__(self, df): #constructor method - no object is calling the constructor method
    super().__init__() #calling the constructor of the parent class
    self.df = df #this is attributes of the object

  def __len__(self): #length method
    return len(self.df) #how many rows are in our dataset?

  def __getitem__(self,i):
    row = self.df.iloc[i] #grabs one row of data
    yi = torch.tensor(row[0])
    xi = torch.tensor(row[1:].values/255.0) #pytorch doesn't want a pandas object
    return xi, yi 
    #not exactly what we want to return, but close
    #pytorch doesn't want to work with a pandas structure

my_dataset = DigitsDataset(df_train)

xi, yi = my_dataset.__getitem__(5)

N_train = df_train.shape[0]
i = np.random.randint(N_train) #this gives us a different number every time
xi, yi = my_dataset.__getitem__(i)
plt.figure()
plt.imshow(torch.reshape(xi, (28,28)), cmap ='gray')
plt.title(str(yi))

#Now that we've created our dataset object, we need to create a data loader
#a data loader grabs batches of data

dataset_train = DigitsDataset(df_train)
dataset_val = DigitsDataset(df_val)

data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 64)

x_batch, y_batch = next(iter(data_loader_train))

type(x_batch)

