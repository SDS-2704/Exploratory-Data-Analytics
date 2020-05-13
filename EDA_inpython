#!/usr/bin/env python
# coding: utf-8

# # EXPLORATORY DATA ANALYSIS

# ### Exploratory Data Analysis (prominently referred to as EDA in the Data Analytics & Data Science industry), is a statistical term that basically defines the approach to analysing data sets to find patterns and relations in your dataset.

# ### Before any kind of machine learning model is created, it is very important for one to understand his/her dataset, evaluate it, deep dive into it and draw productive insights.

# In[58]:


#Just creating a class to use these colors in print statements if required later.
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# ### It is always a good idea to explore the data first and understand it by drawing insights. Now, to perform EDA, I would need a sample dataset. 

# However, before we begin with any kind of statistical or algorthmic operations with Python, it is important to import the necessary libraries. Some of the important libraries we will need are :-
#     
# 
#     1) NumPy - It is a Python package that stands for "Numerical Python". It is the main library in Python which is used by the Data Science & Machine Learning community around the world to do scientific computing. It is based on the usage of powerful n-dimensional array objects or matrices. It also has its scope in linear algebra, and other mathematical disciplines.
#     
#     2) Pandas - It is a high level data manipulation tool that is built on the NumPy package. Its key data structure is called a DataFrame. Data Frames basically allow you to store tabular raw data available in the form of rows of observations and columns of variables.
#     
#     3) Keras - It is an open-source DL neural network library in Python that can run on top of Tensor Flow, Theano, etc.
#     
#     4) matplotlib - It is a Data visualization library of Python. It is used to create plots basis analytical and statistical inferences from the data.

# In[95]:


#!pip install keras

import numpy as np
import pandas as pd
#import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#import tensorflow as tf
import pylab as pl


# # Loading data available online

# ### If I want to upload a dataset that is available online, I can make use of libraries in Python like "Keras". For example, below I am going to load a dataset called fashion_mnist from the Keras library.
# fashion_mnist is a very famous Zolando's (a popular European e-comm company that deals in fashionware / clothing) article images dataset. 
# It has got a training set of 60,000 images and a testing set of 10,000 images.

# # Let's do it on GOOGLE COLAB as it requires GPU

# In[13]:


# # Loading data from Keras library
# fashion_mnist = keras.datasets.fashion_mnist
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# print('X_train shape:', X_train.shape)
# print('y_train shape:', y_train.shape)
# print('X_test shape:', X_test.shape)
# print('y_test shape:', y_test.shape)
# print(X_train.shape[1])
# #print(y_train.shape[1])
# print(X_test.shape[1])
# #print(y_test.shape[1])


# In[14]:


# #Just to see some sample visualizations on the images data using matplotlib
# import matplotlib.pyplot as plt
# image_index = 7707 # You may select anything up to 60,000
# print(y_train[image_index]) # The label is a t-shirt
# plt.imshow(X_train[image_index], cmap='Greys')


# # Another way of uploading data is from your local system, supposedly the easier way.

# #### I have a dataset on Roger Federer, installed in my local system, and I shall upload it from there.

# In[60]:


roger_data = pd.read_csv(r'G:\G_April 30 2020\April 30 2020\Roger Federer Machine Learning Dataset\data\Roger-Federer.csv')


# ### Now, I will create a data frame of the data we just downloaded.

# In[61]:


roger_data_df = pd.DataFrame(roger_data)


# ### If I just want to see the first five or last five records of the data-frame, I can employ head and tail functions respectively.

# In[62]:


roger_data_df.head(5)


# In[63]:


roger_data_df.tail(2)


# ### Now, I would want to know how many rows and columns are there in this dataframe :-

# In[64]:


roger_data_df.shape


# ### Now, it is also a good practice to know the columns and their respective data types in your dataset. We should also look at whether any of those columns have null values or not.

# In[65]:


roger_data_df.info()


# ### Now, I might also want to get a little deeper into the data and see various summary statistics like mean, median, quantile, etc.

# In[66]:


roger_data_df.describe()


# In[67]:


roger_data_df['round'].unique()


# In[68]:


print(color.BLUE + color.BOLD + "Total values in 'round' colum" + color.END)
print(roger_data_df['round'].count())

print("\n" + color.RED + color.BOLD + "Total unique values count in 'round' column" + color.END)
print(roger_data_df['round'].value_counts())


# ### Let's do some duplicate and null value analysis :-

# In[69]:


#Here, am going to look at the rows containing duplicate data
duplicate_rows_df = roger_data_df[roger_data_df.duplicated()]
print(color.BOLD + color.PURPLE + "\nNumber of duplicate rows, columns in the the dataset is " + color.END, duplicate_rows_df.shape, "\n")
#A\ns + color.END it is quite clear now from the information above that there is no duplicate records (rows) as such, hence we will move forward now.
print(color.BOLD + color.GREEN + "\nAs it is quite clear now from the information above that there is no duplicate records (rows) as such, hence we will move forward now.\n" + color.END)
#Here, am going to do the null value analysis. 
print(color.BOLD + color.YELLOW + "\nNumber of null values rows OR no. of null values in each column of the the dataset:-\n" + color.END)
print(roger_data_df.isnull().sum())
print(color.BOLD + color.RED + "\nThe columns in the dataset that have null/missing values:-\n")
roger_data_df.columns[roger_data_df.isnull().any()]

#Let's create a new dataframe with only those columns from the dataframe that have null values
print(color.BOLD + color.GREEN + "\nLet's create a new dataframe with only those columns from the dataframe that have null values:-\n" + color.END)
roger_data_null_df = roger_data_df[roger_data_df.columns[roger_data_df.isnull().any()]].head(5)
roger_data_null_df.head(5)


# ### Looking at only the numeric columns :-

# In[48]:


#Let's look at only the numeric columns from the dataset. The reasons being that we will have to create a new dataframe with only the numeric columns so as to be able to draw plots and be able to impute their missing values with the relevant strategy like mean. 

print(color.BOLD + color.BLUE + "\nLet's look at only the numeric columns from the dataset that have null values. The reasons being that we will have to create a new dataframe with only the numeric columns so as to be able to draw plots and be able to impute their missing values with the relevant strategy like mean." + color.END)

numrc_null_numrc_cols = roger_data_null_df.select_dtypes(include=np.number).columns.tolist()
numrc_null_numrc_cols


# ### Null Value Analysis & Imputations :-

# In[74]:


from sklearn.preprocessing import Imputer
#Here, am going to impute the missing values with mean. The two reasons of imputing the missing values with mean and not dropping them is that the number is reasnably high of the missing values and also secondly, the columns that have the missing values are not necessarily going to be used in further analysis because they are service points / games level data.
print(color.BOLD + color.GREEN + "\nHere, am going to impute the missing values with mean. The two reasons of imputing the missing values with mean and not dropping them is that the number is reasnably high of the missing values and also secondly, the columns that have the missing values are not necessarily going to be used in further analysis because they are service points / games level data.\n" + color.END)


# Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# Train the imputor on the df dataset
mean_imputer = mean_imputer.fit(roger_data_df[numrc_null_numrc_cols])
roger_data_imputed = mean_imputer.transform(roger_data_df[numrc_null_numrc_cols])
roger_data_df[numrc_null_numrc_cols] = roger_data_imputed

#Here, am going to do the null value analysis again mean-imputation.
print(color.BOLD + color.YELLOW + "\nNumber of null values rows OR no. of null values in each column of the the dataset after null values imputation:-\n" + color.END)
print(roger_data_df.isnull().sum())
print(color.BOLD + color.RED + "\nThe columns in the dataset that have null/missing values now:-\n")
roger_data_df.columns[roger_data_df.isnull().any()]


# In[50]:


# # Dropping irrelevant columns
# roger_data_df = roger_data_df.drop(['Column 1', 'Column 2', 'Column 3'], axis=1)
# roger_data_df.head(5)

# #I am not going to execute these lines of code as of now and wait for later when I see that certain columns are not really required.


# # Let's do some Data Visualizations :-

# In[78]:


fig = roger_data_df.hist('year', xlabelsize = 34, ylabelsize = 18, xrot = 45, grid=True, figsize=(8,6), color='#86bf91', zorder=1, rwidth=0.95)
pl.title("Total games played by Federer every year b/w 1998-2012")
pl.xlabel("Year")
pl.ylabel("Matches Played")
pl.show(fig)


# In[54]:


# roger_data_df['tournament prize money (dollars)'] = roger_data_df['tournament prize money (dollars)'].fillna(0)
# roger_data_df['tournament prize money (dollars)'].astype(int)


                                      


# In[ ]:


# roger_data_df['tournament prize money (dollars)'] = pd.to_numeric(roger_data_df['tournament prize money (dollars)'])


# In[ ]:


# roger_data_df.groupby(['year']).sum()['tournament prize money (dollars)']


# In[96]:


#fig = roger_data_df.groupby('year').hist('year', xlabelsize = 18, ylabelsize = 18, xrot = 45, grid=False, figsize=(10,8), color='#86bf91', zorder=1, rwidth=0.95)

fig, ax = plt.subplots(figsize=(15,7))

scale_factor = 0.2 #to change the range of the values on yaxis, one can mulitply the values by a scale factor.
plt.ylim(0 * scale_factor, 385788608 * scale_factor)
roger_data_df.groupby(['year']).sum()['tournament prize money (dollars)'].plot(ax=ax)





pl.title("Prize money earned by Federer every year b/w 1998-2012")
pl.xlabel("Year")
pl.ylabel("Prize Money")
pl.show()


# # Writing data to a csv file

# In[83]:


import pandas as pd
sds = [['a', 1], ['b', 2], ['c', 3]]
df = pd.DataFrame(sds, columns = ['Column1', 'Column2'])
df


# In[84]:



df.to_csv(r'C:\Users\my\Desktop\Sunday Data Analytics _ Presentation\sample_csv\helloeveryone.csv')


# In[15]:


sds_dict = {'column1' : ['a', 'b', 'c'], 'column2' : [1,2,3]}
df_2 = pd.DataFrame(sds_dict)


# In[17]:


df_2


# In[85]:


import requests as req
from bs4 import BeautifulSoup as BS
page = req.get("https://www.amrapali.ac.in/")


# # Creating an object of BeautifulSoup and using html.parser to parse the HTML retrieved from page.

# In[87]:


soup = BS(page.text, 'html.parser')


# In[88]:


soup #let's look at how the HTML is laid/structured on our website.


# # I want to know what all departments are there in Amrapali Institute

# In[89]:


import re
find_ = soup.find_all(class_ = "dropdown-menu")
list_ = []
for line in find_:
    lines_ = str(line).split("</li>") 
    for line_line in lines_:
        if 'Faculty' in line_line:
            pattern = "(Faculty.+)\</a>"
            sub_str_ = re.search(pattern, str(line_line)).group(1).replace('&amp;', '&')
            print(sub_str_)
# #     fac_ = line.find('about.+')
# #     print(fac_)


# # I want to download all the images available on Amrapali's website

# In[93]:


import os



#save_path = 'C:\Users\my\Desktop\Distinguished Alum _ AGI'

import re
import requests
from bs4 import BeautifulSoup

site = 'https://www.amrapali.ac.in/'

response = requests.get(site)

soup = BeautifulSoup(response.text, 'html.parser')
img_tags = soup.find_all('img')

urls = [img['src'] for img in img_tags]


for url in urls:
    filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
    if not filename:
         print("Regex didn't match with the url: {}".format(url))
         continue
    path = r'C:\Users\my\Desktop\Distinguished Alum _ AGI'
    fullpath = os.path.join(path, filename.group(1))
    with open(fullpath, 'wb') as f:
        
        path = r'C:\Users\my\Desktop\Distinguished Alum _ AGI'
        fullpath = os.path.join(path, filename.group(1))
        if 'http' not in url:
            # sometimes an image source can be relative 
            # if it is provide the base url which also happens 
            # to be the site variable atm. 
            url = '{}{}'.format(site, url)
        response = requests.get(url)
        f.write(response.content)


# In[92]:


pwd()


# In[ ]:





# In[ ]:




