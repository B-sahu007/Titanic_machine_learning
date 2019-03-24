
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as py 
import matplotlib.pyplot as plt

import seaborn as sns


test = pd.read_csv('/Users/billu/Downloads/all/test.csv')
test.head()


# In[2]:


train = pd.read_csv('/Users/billu/Downloads/all/train.csv')
train.head()


# In[3]:


train.describe()


# In[4]:


test.describe()


# In[5]:


test.describe(include=['O'])


# In[6]:


train.info()


# In[7]:


train.describe(include=['O'])


# In[8]:


test.info()


# In[9]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))


# In[10]:


train.Pclass.value_counts()


# In[11]:


train.groupby('Pclass').Survived.value_counts()


# In[12]:


train.groupby('Survived').Pclass.value_counts()


# In[13]:


train.mean()  #data frame nhi h #


# In[14]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() #data frame h #


# In[15]:


train[['Pclass', 'Survived']].groupby('Pclass').mean()


# In[16]:


train[['Pclass', 'Survived']].groupby(['Pclass'] , as_index=True).mean()


# In[17]:


train[['Survived','Pclass']].groupby('Pclass').mean()


# In[18]:


train.groupby('Pclass').max()


# In[19]:


#graph plotting#

sns.barplot(x='Pclass' , y='Survived', data= train)


# In[20]:


#sex vs survival#

train.Sex.value_counts()


# In[21]:


train.groupby('Sex').Survived.value_counts()


# In[22]:


train.groupby('Sex').Survived.value_counts()


# In[23]:


sns.barplot(x='Sex',y='Survived',data = train)


# In[24]:


x_axis = train.index
print(x_axis)
y_axis = train['Survived']
plt.bar(x_axis,y_axis,color='r',alpha = 0.5)


# In[25]:


x_axis = train.index
print(x_axis)
y_axis = train['Survived']
plt.bar(x_axis,y_axis,color='b',alpha = 0.8)


# In[26]:


train[['Sex','Pclass','Survived']].groupby('Pclass')


# In[27]:


#use of cross tab #
tab = pd.crosstab(train['Pclass'],train['Sex'])
print(tab)


# In[28]:


tab.div(tab.sum(1).astype(float),axis = 0).plot(kind = "bar")


# In[29]:


tab = pd.crosstab(train['Sex'],train['Pclass'])
print(tab)


# In[30]:


tab.plot(kind = "bar",stacked = True)


# In[31]:


#note in cross tab we can access the value_counts ,whereas in group by we cant #


# In[32]:


sns.factorplot('Pclass','Survived',hue ='Sex',data = train,size = 6,aspect = 2)


# In[33]:


sns.factorplot('Sex','Survived',hue ='Pclass',data = train,size = 6,aspect = 2)


# In[34]:


sns.factorplot('Pclass','Sex',hue = 'Survived',data=train,size =6,aspect=2)


# In[35]:


sns.factorplot('Pclass','Survived',hue = 'Sex',col ='Embarked',data =train)


# In[36]:


theta=[[0],[0]]
for i in range(100):
    j=0
    k=1
    theta[0][0]=theta[j][0]-0.1*2*(theta[j][0]-5)
    theta[1][0]=theta[k][0]-0.1*2*(theta[k][0]-5)
    
    
print(theta[0][0])
print(theta[1][0])


# In[37]:


train.Embarked.value_counts()


# In[38]:


train.groupby('Embarked').Survived.value_counts()


# In[39]:


train.groupby('Embarked').Sex.value_counts()


# In[40]:


train[['Embarked','Survived']].groupby('Embarked').mean()


# In[41]:


df=pd.crosstab(train['Embarked'],train['Survived'])


# In[42]:


print(df)


# In[43]:


df.div(df.sum(1).astype(float),axis =0).plot(kind ='bar',stacked ='True')


# In[44]:


sns.factorplot('Embarked','Survived',hue ='Sex',data = train,aspect = 1,size=6)


# In[45]:


sns.barplot('Embarked','Survived',data=train)


# In[46]:


train.Parch.value_counts()


# In[47]:


train.groupby('Parch').Survived.value_counts()


# In[48]:


sns.barplot('Parch','Survived',data=train)


# In[49]:


#Sibsp vs Survival#


# In[50]:


train.SibSp.value_counts()


# In[51]:





train.groupby('SibSp').Survived.value_counts()


# In[52]:


sns.barplot('SibSp','Survived',ci =None ,data=train)


# In[53]:


#Age vs Suvival#


# In[54]:


fig =plt.figure(figsize=(20,20))
ax1=fig.add_subplot(441)
ax2=fig.add_subplot(442)
ax3=fig.add_subplot(443)
ax4=fig.add_subplot(444)
ax5=fig.add_subplot(445)
ax6=fig.add_subplot(446)
ax7=fig.add_subplot(447)
ax8=fig.add_subplot(448)
ax9=fig.add_subplot(449)


sns.barplot('SibSp','Survived',ci =None ,data=train,ax=ax1)
sns.barplot('Parch','Survived',data=train,ax =ax2,ci=None)
sns.barplot('Embarked','Survived',data=train,ci=None,ax=ax3)
sns.violinplot('Embarked','Age',hue ='Survived',data=train,ax=ax4,split=True)
sns.violinplot('Pclass','Age',hue='Survived',data=train,ax=ax5,split=True)
sns.violinplot('Sex','Age',hue='Survived',data=train,ax=ax6,split=True)





# In[55]:



sns.violinplot('Embarked','Survived',ci=None,data=train,split=True)


# In[56]:


total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]


print(total_survived.shape)
print(total_not_survived.shape)
print(male_survived.shape)
print(male_not_survived.shape)
print(female_survived.shape) 




# In[57]:


plt.figure(figsize =(10,10))
plt.subplot(111)

sns.barplot(x=total_survived.index , y=total_survived['Age'],ci =None,data=train)
sns


# In[58]:


x= total_survived.index
y=total_survived['Age']

plt.scatter(x,y)


# In[70]:


plt.figure(figsize =(15,5))
plt.subplot(111)
plt.subplot(123)

sns.distplot(total_survived['Age'].dropna().values,bins=range(0,81,1),kde =True,color='blue')
sns.distplot(total_not_survived['Age'].dropna().values,bins=range(0,81,1),kde=True,color='red')


# In[60]:


plt.figure(figsize = [10,10])



plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values,bins =range(0,81,1),color='green',kde=True)
sns.distplot(female_not_survived['Age'].dropna().values,bins=range(0,81,1),color='blue',kde=True)


plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values,bins =range(0,81,1),color='green',kde=True)
sns.distplot(male_not_survived['Age'].dropna().values,bins=range(0,81,1),color='blue',kde=True)





# In[61]:


#heatmap#


# In[62]:


plt.figure(figsize=[10,10])
sns.heatmap(train.drop('PassengerId',axis=1).corr(),vmax=0.6,square=True,annot=True)


# In[63]:


#lode heat map fir s krna h #


# # FEATURE EXTRACTION

# In[64]:


train_test_data = [train, test] 


# In[65]:



test.head()


# In[66]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')


# In[67]:


pd.crosstab(train['Title'],train['Sex'])


# In[68]:


for data in train_test_data:
    data['Title']=data['Title'].replace(['Capt','Col','Don','Countess','Dr','Jonkheer','Lady','Major','Rev','Sir'],'Other')
    
    
    data['Title']=data['Title'].replace('Mlle','Miss')
    
    data['Title']=data['Title'].replace('Mme','Mrs')
    data['Title']=data['Title'].replace('Ms','Miss')
    


# In[69]:


train[['Title','Survived']].groupby('Title').mean()


# In[70]:



#one step can be avoided directly using mapping to Capt,Col,Don.....extra to the numeric value thus creating a mapping function #


# In[71]:


title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Other":5}
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)


# In[72]:


train.head(8)


# In[73]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'female': '1', 'male': '0'})


# In[74]:


train.head(10)


# In[75]:


train.Embarked.value_counts()


# In[76]:


for data in train_test_data:
    data['Embarked'] = data['Embarked'].fillna('S') 


# In[77]:


train.Embarked.value_counts()


# In[78]:


#changing Embarkedin numeric values #
NUMBERS = {"S":0,"C":1,"Q":2}
for data in train_test_data:
    data['Embarked']=data['Embarked'].map(NUMBERS)


# In[79]:


train.head()


# In[80]:


train.head()


# In[81]:


for data in train_test_data:
    age_avg = data['Age'].mean()
    age_std = data['Age'].std()
    age_null_count = data['Age'].isnull().sum()
    


# In[82]:


print(train['Age'][2])


# In[83]:


for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    


# In[84]:


#upar wala samjna h tko#


# In[85]:


train.head()


# In[86]:


train['AgeBand'] = pd.cut(train['Age'],5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())


# In[87]:


train.head()


# In[88]:


for data in train_test_data:
    data.loc[(data['Age'] <= 16) ,'Age']= 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32),'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48),'Age'] = 2
    data.loc[(data['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    data.loc[ (data['Age'] > 64), 'Age'] = 4
    


# In[89]:


train.head()


# In[90]:


for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# In[91]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())


# In[92]:


#qcut padna h #


# In[93]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[94]:


train.head()


# In[95]:


#SibSp and Parch feature#


# In[96]:


for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# In[97]:


for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# # FEATURE DROP

# In[98]:


features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop,axis = 1 )
test = test.drop(features_drop ,axis = 1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'],axis = 1)


# In[99]:


train.head()


# In[100]:


test.head()


# In[104]:


X_train = train.drop('Survived',axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId",axis=1).copy()


X_train.shape, y_train.shape, X_test.shape


# In[105]:


#Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# # LOGISTIC REGRESSION

# In[106]:


from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)

