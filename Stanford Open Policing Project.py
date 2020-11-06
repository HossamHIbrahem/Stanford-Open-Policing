#!/usr/bin/env python
# coding: utf-8

# # Dataset: Stanford Open Policing Project

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ri = pd.read_csv('police.csv')


# In[3]:


ri.head()


# In[4]:


ri.shape

ri.dtypes
# In[5]:


ri.isnull().sum()


# In[6]:


#Remove the column that only contains missing values
ri.drop('county_name', axis='columns', inplace = True)


# In[7]:


ri.shape


# In[8]:


ri.columns


# # 2. Do men or women speed more often?
# 
# 

# In[9]:


# when someone is stopped for speeding, how often is it a man or woman?
ri[ri.violation == 'Speeding'].driver_gender.value_counts(normalize = True)


# In[10]:


# when a man is pulled over, how often is it for speeding?
ri[ri.driver_gender == 'M'].violation.value_counts(normalize = True)


# In[11]:


# when a woman is pulled over, how often is it for speeding?
ri[ri.driver_gender == 'F'].violation.value_counts(normalize = True)


# In[12]:


# combines the two lines above
ri.groupby('driver_gender').violation.value_counts(normalize=True)


# # 3. Does gender affect who gets searched during a stop?

# In[13]:


ri.groupby(['violation','driver_gender']).search_conducted.mean()


# 
# # 4. Why is search_type missing so often? 
# 

# In[14]:


ri.isnull().sum()


# In[15]:


ri.search_conducted.value_counts()


# In[16]:


## value_counts ignores missing values by default
ri[ri.search_conducted == False].search_type.value_counts(dropna=False)


# In[17]:


# when search_conducted is True, search_type is never missing
ri[ri.search_conducted == True].search_type.value_counts(dropna=False)


# In[18]:


ri[ri.search_conducted == True].search_type.isnull().sum()


# # 5. During a search, how often is the driver frisked? 
# 

# In[19]:


ri['frisk'] = ri.search_type.str.contains('Protective Frisk')


# In[21]:


ri.frisk.value_counts(dropna = False)


# In[23]:


ri.frisk.mean()


# # 6. Which year had the least number of stops?

# In[24]:


ri.stop_date.str.slice(0, 4).value_counts()


# In[25]:


combined = ri.stop_date.str.cat(ri.stop_time, sep = ' ')


# In[28]:


ri['stop_datetime'] = pd.to_datetime(combined)


# In[33]:


ri.stop_datetime.dt.year.value_counts().sort_values().index[0]


# # 7. How does drug activity change by time of day? 

# In[36]:


ri.drugs_related_stop.dtype


# In[39]:


ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean()


# In[40]:


ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean().plot()


# In[41]:


ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.sum().plot()


# # 8. Do most stops occur at night?

# In[42]:


ri.stop_datetime.dt.hour.value_counts()


# In[43]:


ri.stop_datetime.dt.hour.value_counts().plot()


# In[44]:


ri.stop_datetime.dt.hour.value_counts().sort_index().plot()


# In[47]:


# another method
ri.groupby(ri.stop_datetime.dt.hour).stop_date.count().plot()


# # 9. Find the bad data in the stop_duration column and fix it

# In[48]:


ri.stop_duration.value_counts()


# In[52]:


# what two things are still wrong with this code?
ri[(ri.stop_duration == '1')| (ri.stop_duration == '2')].stop_duration = 'NaN'


# In[53]:


ri.stop_duration.value_counts()


# In[56]:


ri.stop_duration.value_counts(dropna=False)


# In[57]:


ri.loc[ri.stop_duration == 'NaN', 'stop_duration'] = np.nan


# In[58]:


ri.stop_duration.value_counts(dropna=False)


# In[59]:


# another method
ri.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)


# # 10. What is the mean stop_duration for each violation_raw?

# In[60]:


# make sure you create this column
mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
ri['stop_minutes'] = ri.stop_duration.map(mapping)


# In[61]:


ri.stop_minutes.value_counts()


# In[62]:


ri.groupby('violation_raw').stop_minutes.mean()


# In[64]:


ri.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])


# In[ ]:




