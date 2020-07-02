#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
# Local directory
Reviewdata = pd.read_csv('train_data.csv')
#data taken from kaggle
Reviewdata.columns


# In[5]:


### Checking for the Distribution of Default ###
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print('Percentage for default\n')
print(round(Reviewdata.type.value_counts(normalize=True)*100,2))
round(Reviewdata.type.value_counts(normalize=True)*100,2).plot(kind='bar')
plt.title('Percentage Distributions by review type')
plt.show()


# In[6]:


# Apply first level cleaning
import re
import string

#This function converts to lower-case, removes square bracket, removes numbers and punctuation
def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
cleaned1 = lambda x: text_clean_1(x)


# In[8]:


# Let's take a look at the updated text
Reviewdata['cleaned_description'] = pd.DataFrame(Reviewdata.review.apply(cleaned1))
Reviewdata.head(10)


# In[9]:


# Apply a second round of cleaning
def text_clean_2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

cleaned2 = lambda x: text_clean_2(x)

# Let's take a look at the updated text
Reviewdata['cleaned_description_new'] = pd.DataFrame(Reviewdata['cleaned_description'].apply(cleaned2))
Reviewdata.head(10)


# In[12]:


Reviewdata.drop(columns = ['review'], inplace = True)
Reviewdata.head(4)


# In[14]:


from sklearn.model_selection import train_test_split
Independent_var = Reviewdata.cleaned_description_new
Dependent_var = Reviewdata.type
IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size = 0.2, random_state = 225)
print('IV_train :', len(IV_train))
print('IV_test  :', len(IV_test))
print('DV_train :', len(DV_train))
print('DV_test  :', len(DV_test))


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec = TfidfVectorizer()

clf2 = LogisticRegression(max_iter=1000) 


# In[16]:


from sklearn.pipeline import Pipeline


model = Pipeline([('vectorizer',tvec),('classifier',clf2)])

model.fit(IV_train, DV_train)


# In[17]:


from sklearn.metrics import confusion_matrix

predictions = model.predict(IV_test)

confusion_matrix(predictions, DV_test)


# In[18]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Accuracy : ", accuracy_score(predictions, DV_test))
print("Precision : ", precision_score(predictions, DV_test, average = 'weighted'))
print("Recall : ", recall_score(predictions, DV_test, average = 'weighted'))


# In[31]:


ex=[input(("enter a string: "))]
n=model.predict(ex)
if(n==0):
    print("negative")
elif (n==1):
    print("positive")


# In[30]:


ex=[input(("enter a string: "))]
n=model.predict(ex)
if(n==0):
    print("negative")
elif (n==1):
    print("positive")


# In[ ]:




