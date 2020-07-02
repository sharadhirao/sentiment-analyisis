#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd 
# Local directory
Reviewdata = pd.read_csv('train_data.csv')
#data taken from kaggle


# In[38]:


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


# In[39]:


Reviewdata.columns


# In[40]:


Reviewdata['cleaned_description'] = pd.DataFrame(Reviewdata.review.apply(cleaned1))
Reviewdata.head(5)


# In[41]:


# Apply a second round of cleaning
def text_clean_2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

cleaned2 = lambda x: text_clean_2(x)

# Let's take a look at the updated text
Reviewdata['cleaned_description_new'] = pd.DataFrame(Reviewdata['cleaned_description'].apply(cleaned2))
Reviewdata.head(5)


# In[42]:


#remove unnecessary columns
Reviewdata.drop(columns = ['review','cleaned_description'], inplace = True)
Reviewdata.head(4)


# In[43]:


from sklearn.model_selection import train_test_split
Independent_var = Reviewdata.cleaned_description_new
Dependent_var = Reviewdata.type
IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size = 0.2, random_state = 225)


# In[44]:


#vectorizeing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

tvec = TfidfVectorizer()

clf2 = RandomForestClassifier()


# In[45]:


#using pipeline pass data to ran
from sklearn.pipeline import Pipeline

model = Pipeline([('vectorizer',tvec),('classifier',clf2)])

model.fit(IV_train, DV_train)


# In[46]:


from sklearn.metrics import confusion_matrix

predictions = model.predict(IV_test)

confusion_matrix(predictions, DV_test)


# In[47]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Accuracy : ", accuracy_score(predictions, DV_test))
print("Precision : ", precision_score(predictions, DV_test, average = 'weighted'))
print("Recall : ", recall_score(predictions, DV_test, average = 'weighted'))


# In[51]:


ex=[input(("enter a string: "))]
data=model.predict(ex)
if(data==0):
    print("negative review")
elif data==1:
    print("positive review")


# In[52]:


ex=[input(("enter a string: "))]
data=model.predict(ex)
if(data==0):
    print("negative review")
elif data==1:
    print("positive review")


# In[ ]:




