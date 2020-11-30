#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Importing Dataset
# 1. Since the dataset in excel file we have to use pandas read_excel to load the data
# 2.after loading it is important to check the complete information of data as it can be indication many of the hidden information such as null value in a column and row.
# 3.check wweather any null values are there or not,if it present the take folling step:
#   a.Imputing data using imputation method sklearn
#   b.Filling NaN values with mean,meadian and mode using fillna() method
# 4.Describe data---which can give some statical analysis

# In[2]:


df=pd.read_excel(r"D:\Flight Fare Predictions\Data_train.xlsx")


# In[3]:


pd.set_option("display.max_columns",None)


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df['Duration'].value_counts()


# In[8]:


df.dropna(inplace=True)  #drop the rows which NaN  values


# In[9]:


df.isnull().sum()


# In[10]:


df.shape


# # EDA
# from description we see that the Date_of_journey is a object data type,therefore we need to convert this datatype into timestamp so as to use this column properly for prediction ,for this we need pandas to_datetime to convert object data type to datetime dtype.
# #.dt.day method will exactly only day of that date
# #.da.month method will extract only month of that date

# In[11]:


df["Journey_day"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day


# In[12]:


df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format = "%d/%m/%Y").dt.month


# In[13]:


df.head()


# In[14]:


df.drop(columns=["Date_of_Journey"],inplace=True)  #we are droping Date_of_Journey columns


# In[15]:


#Departure time is when the a plane leaves the Gate.
#Similar to Date_of_Journey we can extract value from Dep_Time


#Extracting Hours
df["Dep_hour"]=pd.to_datetime(df["Dep_Time"]).dt.hour

#Extracting Minutes
df["Dep_min"]=pd.to_datetime(df["Dep_Time"]).dt.minute

##now we can drop Dep_Time

df.drop(columns=["Dep_Time"],inplace=True)


# In[16]:


df.head()


# In[17]:


#Arrival time is when the a plane arrives the Gate.
#Similar to Date_of_Journey we can extract value from Arrival_Time


#Extracting Hours
df["Arrival_hour"]=pd.to_datetime(df["Arrival_Time"]).dt.hour

#Extracting Minutes
df["Arrival_min"]=pd.to_datetime(df["Arrival_Time"]).dt.minute

##now we can drop Dep_Time

df.drop(columns=["Arrival_Time"],inplace=True)


# In[18]:


df.head()


# #Time taken by plane to reach from source to destination airpot called Duration
# #it is also called diff between arrival time and departure time
# 
# #Assigninig and converting Duration column into list
# duration=list(df["Duration"])
# 
# for i in range (len(duration)):
#     if len(duration[i].split()) != 2:  #check wheather duration contains only hours or mins
#         if "h" in duration[i]:
#             duration[i]=duration[i].strip() + "0m"  #Add 0 min
#         else:
#             duration[i]="0h" + duration[i]      #Add 0 hours
#             
# duration_hours=[]
# duration_mins=[]
# for i in range (len(duration)):
#     duration_hours.append(int(duration[i].split(sep="h")[0]))  #Extract hours from duration
#     duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))  #Extract only minutes from duration

# In[19]:


len("2h 50min".split())  #for Information about splitting 


# In[20]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[21]:


# Adding duration_hours and duration_mins list to train_data dataframe

df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins


# In[22]:


df.drop(["Duration"],axis=1,inplace=True)


# In[23]:


df.head()


# #Handling Categorical Data
# 
# #we have many way to find handling categorical data.some of them categorical data are
# #1.Nominal data--data are not in order---->OneHot Encoading is used in this case(not comparable,not indexing)
# #2.Ordinal data--.data are in order--->Label Encoader is used in this case

# In[24]:


df["Airline"].value_counts()


# In[25]:


#From graph we can see that the Jet Airways Bussiness have the Higher Price
#Apart from the first Airline almost all are having similar median


#Airline VS Price
sns.catplot(y = "Price", x = "Airline", data = df.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[26]:


#As Airline is Nominal categorical data we will perform OneHotEncoding
Airline= df[["Airline"]]

Airline= pd.get_dummies(Airline ,drop_first=True)

Airline.head()


# In[27]:


df["Source"].value_counts()


# In[28]:


#Source vs Price

sns.catplot(y = "Price", x = "Source", data = df.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)
plt.show()


# In[29]:


#As SOurce is also a Nominal Categorical data so we use OneHotEncodinng

Source=df[["Source"]]

Source=pd.get_dummies(Source,drop_first=True)

Source.head()


# In[30]:


df["Destination"].value_counts()


# In[31]:


#As Destination is also a Nominal Categorical data so we use OneHotEncodinng

Destination=df[["Destination"]]

Destination=pd.get_dummies(Destination,drop_first=True)

Destination.head()


# In[32]:


df["Route"]


# In[33]:


#Additional value contaional 80% are no_info
#Route and Total_stops are related to each other

df.drop(columns=["Route","Additional_Info"],inplace=True)


# In[34]:


df.head()


# In[35]:


df["Total_Stops"].value_counts()


# In[36]:


#As stops is the case of Ordinal Categorical type we perform LabelEncoder
#Here value are assigned with crossponding keys

df.replace({"non-stop" : 0 , "1 stop" : 1 , "2 stops" : 2, "3 stops" : 3 , "4 stops" : 4},inplace=True)


# In[37]:


df.head()


# In[38]:


#Now we concatenate Dataframe-->df + Airline + Source + Destination

df = pd.concat([df , Airline , Source , Destination],axis=1)


# In[39]:


df.head()


# In[40]:


df.drop(columns=["Airline","Source","Destination"],inplace=True)


# In[41]:


df.head()


# In[42]:


df.shape


# # Now working on Test Set

# In[43]:


test_data=pd.read_excel(r"D:\Flight Fare Predictions\Test_set.xlsx")


# In[44]:


test_data.head()


# In[45]:


# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[46]:


data_test.head()


# # Feature Selection
# 
# #Finding out the best feature which will contribute and have good relation with target variable.Following are some of the feature selection method
# 
# #1Heatmap
# #2.feature-Importance_
# #3.SelectKBest

# In[47]:


df.shape


# In[48]:


df.columns


# In[49]:


X=df.loc[:, ['Total_Stops','Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]

X.head()


# In[50]:


y=df.iloc[:,1]

y.head()


# In[51]:


#Find the correlation between independent and dependent attributes

plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")

plt.show()


# In[52]:


#import feature using of ExtraTreeRegressor (this is use for finding out our important feature)

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[53]:


print(selection.feature_importances_)


# In[54]:


#plot the graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# # Fitting Model using RandomForest
# 
# #1.split dataset into train and test set in order to predict w.r.t. X_test
# #2.if needed of scaling of Data(scaliing is not donr in RandomForest)
# #3.Import model
# #4.Fit the model
# #5. Predict w.r.t. X
# #6.In regression check RSME score
# #7.plot Graph

# In[55]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[56]:


from sklearn.ensemble import RandomForestRegressor
reg_rf=RandomForestRegressor()
reg_rf.fit(X_train,y_train)


# In[57]:


y_pred=reg_rf.predict(X_test)


# In[58]:


reg_rf.score(X_train,y_train)


# In[59]:


reg_rf.score(X_test,y_test)


# In[60]:


sns.distplot(y_test-y_pred)


# In[61]:


plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[62]:


#Now checking our metrics score

from sklearn import metrics


# In[63]:


print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[64]:


#RMSE /(max(DV)-min(DV))

2093/(max(y)-min(y))


# In[65]:


metrics.r2_score(y_test , y_pred)


# # Hyperparameter Tuning
# 
# #choose following method for hyperparameter tuninig
# 1.RandomizedSearchCV--Fast
# 2.GridSearchCV
# 
# #assign parameter in form of dictionary
# #Fit the model
# #check best parameters and best score

# In[66]:


from sklearn.model_selection import RandomizedSearchCV


# In[67]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[68]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[69]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[70]:


rf_random.fit(X_train,y_train)


# In[71]:


rf_random.best_params_


# In[72]:


prediction = rf_random.predict(X_test)


# In[73]:


plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[74]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[75]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# # Saving the model

# In[76]:


import pickle
# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[77]:


model = open('flight_rf.pkl','rb')
forest = pickle.load(model)


# In[78]:


y_prediction = forest.predict(X_test)


# In[79]:


metrics.r2_score(y_test, y_prediction)


# In[ ]:




