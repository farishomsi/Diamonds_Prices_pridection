
# 1.To take a look at the big picture
# the dims


# ## 2.Get the data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('diamonds.csv')
dimsdf = df.copy()
dims2 = dimsdf.copy()


print(dimsdf.head())
print( dimsdf.info())
dimsdf.describe()

dimsdf.hist(bins=40,figsize=(20,15))


sns.kdeplot(dimsdf['price']) #skwed to the right


sns.kdeplot(dimsdf['depth'])#bell-shaped


# ## 3.Discover and visulize the data to get insight

# Looking for correlations

sns.pairplot(dimsdf)
dimsdf.corr()
dimsdf.corr()['depth'].sort_values(ascending=False)

sns.heatmap(dimsdf.corr(), annot=True)

sns.pairplot(dimsdf,hue='color',x_vars=['carat','price'],y_vars=['price','x','y','z'])

sns.pairplot(dimsdf,hue='cut',x_vars=['carat','price'],
             y_vars=['price','x','y','z'],
             palette='coolwarm')

#sns.pairplot(dimsdf,hue='clarity',x_vars=['carat','price'],y_vars=['price','x','y','z'])

#dimsdf.plot(kind="scatter",x='price',y='z',alpha=0.5)


dimsdf.plot.scatter(x='depth',y='table',cmap='coolwarm')

dimsdf.plot.scatter(x='carat',y='price',c='x',cmap='coolwarm')


#### Categorical attributes visulization

sns.boxplot(x='price',y='cut',data=dimsdf)
sns.boxplot(x='price',y='color',data=dimsdf)
sns.boxplot(x='clarity',y='price',data=dimsdf)

#It seems that VS1 and VS2 affect the Diamond's Price equally

dimsdf['color'].value_counts().plot(kind="bar")

dimsdf['cut'].value_counts().plot(kind="bar")

dimsdf['clarity'].value_counts().plot(kind="bar")


# 4.Prepare data for machine learning algorithms

# Aggregation column and drop 'Unnamed: 0' col


dimsdf['diamond_size'] = dimsdf['x']*dimsdf['y']*dimsdf['z']
dimsdf.drop(['Unnamed: 0','x','y','z'] , axis=1,inplace=True)

dimsdf.head()

# converting categorical attributes into numerical (custom Encoder )


def color_switch(arg):
    if arg == 'D':
         return 1
    elif arg == 'E':
         return 2
    elif arg == 'F':
         return 3
    elif arg == 'G':
         return 4
    elif arg == 'H':
        return 5
    elif arg == 'I':
        return 6
    elif arg == 'J':
        return 7
    else:
        return None


def clarity_switch(arg):
    if arg == 'IF':
         return 1
    elif arg == 'VVS1':
         return 2
    elif arg == 'VVS2':
         return 3
    elif arg == 'VS1':
         return 4
    elif arg == 'VS2':
        return 5
    elif arg == 'SI1':
        return 6
    elif arg == 'SI2':
        return 7
    elif arg == 'I1':
        return 8
    else:
        return None


def cut_switch(arg):
    if arg == 'Ideal':
         return 1
    elif arg == 'Premium':
         return 2
    elif arg == 'Very Good':
         return 3
    elif arg == 'Good':
         return 4
    elif arg == 'Fair':
        return 5
    else:
        return None


dimsdf['cut'] = dimsdf['cut'].apply(cut_switch)
dimsdf['clarity'] = dimsdf['clarity'].apply(clarity_switch)
dimsdf['color'] = dimsdf['color'].apply(color_switch)
dimsdf.head()


# Split the  data into train , test set

dimsdf.columns

x = dimsdf[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'diamond_size']]
y = dimsdf['price']
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)


# 5.Select and train a model
# Train the model , prediction and model evaluation
# Linear regression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
import math

regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pred = regr.predict(x_test)
print("accuracy: "+ str(regr.score(x_train,y_train)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))
print("Root Mean squared error: {}".format(math.sqrt(mean_squared_error(y_test,y_pred))))


# Ridge regression

rig_reg = linear_model.Ridge()
rig_reg.fit(x_train,y_train)
y_pred = rig_reg.predict(x_test)
print("accuracy: "+ str(rig_reg.score(x_train,y_train)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))
print("Root Mean squared error: {}".format(math.sqrt(mean_squared_error(y_test,y_pred))))


# Lasso regression

las_reg = linear_model.Lasso()
las_reg.fit(x_train,y_train)
y_pred = las_reg.predict(x_test)
print("accuracy: "+ str(las_reg.score(x_train,y_train)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))
print("Root Mean squared error: {}".format(math.sqrt(mean_squared_error(y_test,y_pred))))


# Random forest regressor

from sklearn.ensemble import RandomForestRegressor
random_reg = RandomForestRegressor()
random_reg.fit(x_train,y_train)
y_pred = random_reg.predict(x_test)
print("accuracy: "+ str(random_reg.score(x_train,y_train)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))
print("Root Mean squared error: {}".format(math.sqrt(mean_squared_error(y_test,y_pred))))


# Delete Outliers :

dims_out = dimsdf.copy()

dims_out.head()

Q1=dims_out['price'].quantile(0.25)
Q3=dims_out['price'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['price']< Upper_Whisker]

dims_out.shape

Q1=dims_out['depth'].quantile(0.25)
Q3=dims_out['depth'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['depth']< Upper_Whisker]

dims_out.shape

Q1=dims_out['carat'].quantile(0.25)
Q3=dims_out['carat'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['carat']< Upper_Whisker]

dims_out.shape


Q1=dims_out['table'].quantile(0.25)
Q3=dims_out['table'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['table']< Upper_Whisker]

dims_out.shape


Q1=dims_out['diamond_size'].quantile(0.25)
Q3=dims_out['diamond_size'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['diamond_size']< Upper_Whisker]

dims_out.shape


# spliting data

x2 = dims_out[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'diamond_size']]
y2 = dims_out['price']
from sklearn.model_selection import train_test_split
x_train2 , x_test2 , y_train2 , y_test2 = train_test_split(x2,y2,test_size=0.2)


# ### Training the  model after droping outliers
# Linear regression

regr = linear_model.LinearRegression()
regr.fit(x_train2,y_train2)
y_pred = regr.predict(x_test2)
print("accuracy: "+ str(regr.score(x_train2,y_train2)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test2,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test2,y_pred)))


# Ridge regression

rig_reg = linear_model.Ridge()
rig_reg.fit(x_train2,y_train2)
y_pred = rig_reg.predict(x_test2)
print("accuracy: "+ str(rig_reg.score(x_train2,y_train2)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test2,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test2,y_pred)))


# Lasso regression

las_reg = linear_model.Lasso()
las_reg.fit(x_train2,y_train2)
y_pred = las_reg.predict(x_test2)
print("accuracy: "+ str(las_reg.score(x_train2,y_train2)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test2,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test2,y_pred)))

# Random forest regressor
random_reg = RandomForestRegressor()
random_reg.fit(x_train2,y_train2)
y_pred = random_reg.predict(x_test2)
print("accuracy: "+ str(random_reg.score(x_train2,y_train2)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test2,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test2,y_pred)))