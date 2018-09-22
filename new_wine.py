
# coding: utf-8

# In[109]:

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import seaborn as sns


# In[2]:

get_ipython().magic('matplotlib notebook')


# In[3]:

file = pd.read_csv('wine_150k.csv')
file.head()
#type(file)


# In[4]:

file.columns


# In[5]:

file.describe()


# In[6]:

file.dropna(thresh=1, inplace=True)


# In[7]:

file.describe()


# In[8]:

file.head()


# In[9]:

file.drop('region_2', axis=1, inplace=True)
type(file)


# In[10]:

file


# In[11]:

file50 =file[:100000]


# In[12]:

file50.head()


# In[13]:

file50.drop('Unnamed: 0', axis=1, inplace=True)


# In[14]:

file50.head()


# In[15]:

file50.describe()


# In[16]:

file50.dropna(how='any', inplace=True)


# In[17]:

file50.count()


# In[18]:

file50


# In[19]:

file50.describe()


# In[20]:

bins1 = [80,85,90,95,101] 
labels1= ['Bronze','Silver','Gold','Platinum']
file50['category']=file50['points']
file50['category'] = pd.cut(file50['points'], bins=bins1 , labels= labels1)
file50


# In[21]:

# define method - ratings
# create new column - 'ratings'
# input points bin
# if point between 80-85, assign random rating between 1 and 3
# if point between 85-90, assign random rating between 2 and 4
# if point between 90-95, assign random rating between 4 and 5
# if point between 95-100, assign random rating between 4.5 and 5

file50['rating']= file50['points']
file50


# In[35]:

rating_list = []
np.random.seed(9)
for ratings in file50['rating']:
    
    if (ratings<85):
        w_ratings = 1 + (np.random.rand() * (3 - 1))
        w_ratings=round(w_ratings,1)
        rating_list.append(w_ratings)
    
    if ((ratings>=85) & (ratings<=90)):
        w_ratings = 2 + (np.random.rand() * (4 - 2))
        w_ratings=round(w_ratings,1)
        rating_list.append(w_ratings)
    
    if ((ratings >=90) & (ratings<=95)):
        w_ratings = 3 + (np.random.rand() * (5 - 3))
        w_ratings=round(w_ratings,1)
        rating_list.append(w_ratings)
    
    if (ratings>95):
        w_ratings = 4 + (np.random.rand() * (5 - 4))
        w_ratings=round(w_ratings,1)
        rating_list.append(w_ratings)


# In[36]:

rating_list


# In[37]:

label=['wine_ratings']
new_df = pd.DataFrame(rating_list, columns =label)
new_df


# In[38]:

file25=pd.concat([file50,new_df], axis=1)
file25


# In[39]:

file25.drop('rating', axis=1, inplace = True)
file25


# In[40]:

file25.count()


# In[41]:

file25.dropna(how = 'any', inplace=True)


# In[42]:

file25.count()


# In[43]:

file25


# In[44]:

user_id = []
np.random.seed(7)
for ids in file25['description']:
    
    if (len(ids)<25):
        u_ids = np.random.randint(1,50)
        user_id.append(u_ids)
    
    if ((len(ids)>=25) & (len(ids)<50)):
        u_ids = np.random.randint(50,100)
        user_id.append(u_ids)
    
    if ((len(ids)>=50) & (len(ids)>=100)):
        u_ids = np.random.randint(100,300)
        user_id.append(u_ids)
    
    if (len(ids)>100):
        u_ids = np.random.randint(300,500)
        user_id.append(u_ids)


# In[45]:

len(user_id)


# In[46]:

label_df=['user_id']
uid_df = pd.DataFrame(user_id, columns =label_df)
uid_df.head()


# In[47]:

file25=pd.concat([file25,uid_df], axis=1)
file25


# In[48]:

file25.count()


# In[50]:

file25.dropna(how = 'any', inplace=True)
file25.count()


# In[51]:

file25.head()


# In[54]:

file25['user_id'].nunique()


# In[55]:

file25['variety'].nunique()


# In[56]:

file25['country'].nunique()


# In[57]:

file25['category'].nunique()


# In[67]:

file25['description'][400]


# In[90]:

time_df = pd.read_csv('ratings.csv')
time_df.count()


# In[91]:

timestamp = datetime.datetime.fromtimestamp(1462644085)
print(timestamp.strftime('%Y-%m-%d %H:%M:%S'))


# In[94]:

file25=pd.concat([file25,time_df], axis=1)
file25.head()


# In[98]:

file25.drop('timestamp', axis = 1)
file25.head()


# In[103]:

file25.drop('timestamp', axis =1, inplace= True)
#file25=pd.concat(timestampfile25)


# In[104]:

file25.describe()


# In[105]:

file25=pd.concat([file25,time_df], axis=1)


# In[106]:

file25.describe()


# In[107]:

file25.head()


# In[111]:

file25.info()


# ### Where did the NaN values came from

# In[118]:

file25.dropna(how='any', inplace=True)
file25.info()


# In[121]:

#sns.pairplot(file25)


# In[120]:

# Plots


# In[123]:

sns.distplot(file25['points'])


# ### Machine Learning

# In[125]:

file25.corr()


# In[127]:

sns.heatmap(file25.corr(), annot=True)


# ### Why did the heatmap not display?

# In[128]:

file25.columns


# In[130]:

X = file25[['points','wine_ratings']]


# In[131]:

y = file25['price']lm.C


# In[132]:

from sklearn.cross_validation import train_test_split


# In[133]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)


# In[134]:

from sklearn.linear_model import LinearRegression


# In[135]:

lm = LinearRegression()


# In[136]:

lm.fit(X_train,y_train)


# In[137]:

print(lm.intercept_)


# In[138]:

lm.coef_


# In[139]:

cdf = pd.DataFrame(lm.coef_,X.columns,columns= ['Coeff'])
cdf


# In[140]:

predictions= lm.predict(X_test)


# In[141]:

predictions


# In[147]:

plt.scatter(y_test,predictions)
plt.show()


# In[146]:

#sns.distplot((y_test-predictions))


# In[148]:

from sklearn import metrics


# In[150]:

np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[151]:

metrics.mean_squared_error(y_test, predictions)


# In[ ]:



