
# It is project of Used Car Price Prediction so first work to import all necessary library files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import Lasso
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# now read the dataset and include here

car_dataset = pd.read_csv(r"C:\Users\debas\OneDrive\Documents\projectData\Car details v3.csv")

# inspecting some rows and all columns from dataframe

pd.set_option('display.max_column', None)
print(car_dataset.head(5))

# now show size of dataframe

print(car_dataset.shape)

# now it's time to collect information about dataframe

print(car_dataset.info())

# delete column torque
car_dataset.drop(["torque"], axis=1, inplace=True)


# we work  obviously on data type but first, find out here having any missing value and using Seaborn plot a graph

print(car_dataset.isnull().sum())
# sns.heatmap(car_dataset.isnull(), yticklabels=False, cbar=False, cmap='Paired')
# plt.show()

# calculate the car is how years old

car_dataset['years_old'] = 2022-car_dataset.year

# now drop the previous year column

car_dataset.drop(['year'], axis=1, inplace=True)
# print(car_dataset.head())


# Now find out what percentage of missing values

miss_value_per_num = car_dataset.isnull().sum()/car_dataset.shape[0]*100
print(miss_value_per_num)

# Here missing values is less than 20% so now we deal missing values with mean and median and mode
# Here not any missing values of categorical data so we don't use mode


# Missing value imputation-Numerical variable

car_dataset_2 = car_dataset.select_dtypes(include=['int64', 'float64'])
print(car_dataset_2.head(5))

# Now plot heat map
plt.figure(figsize=(16, 9))
sns.heatmap(car_dataset_2.isnull())
plt.show()

# Showing that rows,columns where having not any value
print(car_dataset_2.isnull().sum())
miss_value_var_num = [var for var in car_dataset_2.columns if car_dataset_2[var].isnull().sum() > 0]
print(miss_value_var_num)

# now show data distribution

# plt.figure(figsize=(10, 10))
# sns.set()
# sns.distplot(car_dataset_2[miss_value_var_num], bins=20, kde_kws={'linewidth': 5, 'color': '#DC143C'})
# plt.show()  # when missing value is random than use median and mean. plot graph is skew or asymmetric than use median

car_dataset_2_median = car_dataset_2.fillna(car_dataset_2.median())
print(car_dataset_2_median.isnull().sum())

# compare between two plot one with missing values and one which have not any missing value

# plt.figure(figsize=(10, 10))
# sns.set()
# sns.distplot(car_dataset_2[miss_value_var_num], hist=False, bins=20, kde_kws={'linewidth': 8, 'color': '#DC143C'},label='original')
# sns.distplot(car_dataset_2_median[miss_value_var_num], hist= False, bins=20, kde_kws={'linewidth': 5, 'color': 'green'},label='median')
# plt.legend()
# plt.show()  # having less changes in median and original

# now show here any null value

df_concat = pd.concat([car_dataset_2[miss_value_var_num], car_dataset_2_median[miss_value_var_num]],axis=1)
# pd.set_option('display.max_row',None)
# print(df_concat[df_concat.isnull().any(axis=1)])


# Missing Value imputation - Categorical Variable

car_dataset_3 = car_dataset.select_dtypes(include='object')
print(car_dataset_3.head(5))

# In what column how many value is missing

miss_value_per_cat = car_dataset_3.isnull().mean()*100
print(miss_value_per_cat)

# Now find out columns where having no values in rows

miss_value_var_cat = miss_value_per_cat[miss_value_per_cat > 0].keys()
print(miss_value_var_cat)

# value count in Mileage

print(car_dataset_3['mileage'].value_counts())

# Analysing it execute mode method

print(car_dataset_3['mileage'].mode())
# car_dataset_3_mode = car_dataset_3['mileage'].fillna(car_dataset_3['mileage'].mode()[0])
# print(car_dataset_3_mode)
for var in miss_value_var_cat:
    car_dataset_3[var].fillna(car_dataset_3[var].mode()[0], inplace=True)
    print(var, "=", car_dataset_3[var].mode()[0])

print(car_dataset_3.isnull().sum())

# now plot hist graph
# now check distribution of data

plt.figure(figsize=(16, 9))
for i, var in enumerate(miss_value_var_cat):
    plt.subplot(2, 2, i+1)
    plt.hist(car_dataset_3[var], label='Imput')
    plt.hist(car_dataset[var].dropna(), label='original')
    plt.legend()
plt.show()

# update original dataframe
car_dataset.update(car_dataset_2_median)
car_dataset.update(car_dataset_3)
print(car_dataset.select_dtypes(include=['object', 'int64', 'float64']).isnull().sum())


for i in range(car_dataset.shape[0]):
    car_dataset.at[i, 'mileage(km/kg)'] = car_dataset['mileage'][i].split()[0]
    car_dataset.at[i, 'engine(cc)'] = car_dataset['engine'][i].split()[0]
    car_dataset.at[i, 'power(bhp)'] = car_dataset['max_power'][i].split()[0]
    car_dataset.at[i, 'owner_type'] = car_dataset['owner'][i].split()[0]

# now take a look of updating dataframe
print(car_dataset.head(5))

# now Type casting some column with some valid data type
print("Before typecasting:", car_dataset.info())

car_dataset['mileage(km/kg)'] = car_dataset['mileage(km/kg)'].astype(float)
car_dataset['engine(cc)'] = car_dataset['engine(cc)'].astype(float)
# # car_dataset['power(bhp)'] = car_dataset['power(bhp)'].astype(float)
print("After typecasting:", car_dataset.info())
#
f = 'n'
count = 0
position = []
for i in range(car_dataset.shape[0]):
    if car_dataset['power(bhp)'][i] == 'bhp':
        f = 'y'
        count = count+1
        position.append(i)
print(count)
print(position)

# Now drop that null bhp containing indexes
car_dataset = car_dataset.drop(car_dataset.index[position])

# now reset missing index

car_dataset = car_dataset.reset_index(drop=True)
print(car_dataset.shape)

# now typecasting of power(bhp)

car_dataset['power(bhp)'] = car_dataset['power(bhp)'].astype(float)
print("After typecasting:", car_dataset.info())

# backup the dataframe

backup = car_dataset.copy()

# now time to drop useful columns

car_dataset.drop(["name"], axis=1, inplace=True)
car_dataset.drop(["mileage"], axis=1, inplace=True)
car_dataset.drop(["engine"], axis=1, inplace=True)
car_dataset.drop(["max_power"], axis=1, inplace=True)
car_dataset.drop(["owner"], axis=1, inplace=True)

print(car_dataset.info())

# use one hot encoding on categorical data and  dummy variable (k-1) happen when drop_first= True use dummy variableTrap
# using dummy variable categorical data to numerical format
# create dummy variables
dummy_car_dataset = pd.get_dummies(car_dataset)
# print(dummy_car_dataset.head())

# Now use one hot encoding
# space use for we want numpy array
oh_enc = OneHotEncoder(sparse=False, handle_unknown='ignore' )
oh_enc_array = oh_enc.fit_transform(car_dataset[['fuel', 'seller_type', 'transmission','owner_type']])
print(oh_enc_array)

print(dummy_car_dataset.keys())
oh_enc_df = pd.DataFrame(oh_enc_array, columns=['fuel_CNG', 'fuel_Diesel', 'fuel_LPG','fuel_Petrol', 'seller_type_Dealer', 'seller_type_Individual','seller_type_Trustmark Dealer', 'transmission_Automatic','transmission_Manual', 'owner_type_First', 'owner_type_Fourth','owner_type_Second', 'owner_type_Test', 'owner_type_Third'])
print(oh_enc_df)

# Also we use it
# categorical variable -ordinal variable is owner type we scale it and also order it and Nominal variable we not measure
# label encoding encodes first alphabet format (0, 1, 2)
# Ordinal encoding apply on ordinal categorical variables it is like grade A is 3 , grade B is smaller than A so it is 2


# update it and put in main car dataset dataframe
final_car_dataset = car_dataset.join(oh_enc_df)
print(final_car_dataset)
# now drop the original categorical variable
final_car_dataset.drop(["fuel"], axis=1, inplace=True)
final_car_dataset.drop(["seller_type"], axis=1, inplace=True)
final_car_dataset.drop(["transmission"], axis=1, inplace=True)
final_car_dataset.drop(["owner_type"], axis=1, inplace=True)

print(final_car_dataset.info())

# Now split data into trained and target data and apply Feature Selection - pearson Correlation coefficient
# split into trained and test is a holdout validation approach

x = final_car_dataset.drop('selling_price', axis=1)
y = final_car_dataset['selling_price']
print(x.info())
print(y.value_counts())

# Now separate dataset into train and test & prevent overfeeding part

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape)
print(x_test.shape)

# Using pearson correlation

plt.figure(figsize=(12, 10))
cor = x_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

# With the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                col_name = corr_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr


corr_features = correlation(x_train,0.7)
print(len(set(corr_features)))
print(corr_features)

x_train.drop(corr_features, axis=1, inplace=True)
x_test.drop(corr_features, axis=1, inplace=True)
# print(x_train.info())
# print(x_test.info())


# Hyper parameter tuning -using cross validation technique
# K Fold Cross Validation
# "Model Training using Llinear Regression"

print("....Linear Regression....")
# loading Linear regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(x_train, y_train)

# K-fold cross validation
# importing cross _val_ score function

linear_scores = cross_val_score(linear_reg_model, x_train, y_train, scoring='r2', cv=10)
print(linear_scores)

# print average score of training data
print("Accuracy on Training data:", np.mean(linear_scores))

# print the scores of test data and first work predict

linear_test_data_predict = cross_val_predict(linear_reg_model, x_test, y_test)
linear_scores_test = cross_val_score(linear_reg_model, x_test, y_test, cv=10)
print(linear_scores_test)

print("Accuracy on Testing data:", np.mean(linear_scores_test))

# R square error
linear_error_score = metrics.r2_score(y_test, linear_test_data_predict)
print("R squared error:", linear_error_score)
print("Mean absolute error:", metrics.mean_absolute_error(y_test, linear_test_data_predict))

# visualize the actual prices and predicted prices

plt.scatter(x=y_test, y=linear_test_data_predict, c=y_test, cmap='Paired_r', edgecolors='black')
plt.colorbar()
plt.ylabel("Actual Price")
plt.xlabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
#
# "Model training using Lasso Regression"
print("....Lasso Regression....")
# loading Linear regression model
lasso_reg_model = Lasso(tol=3.576e+11)
lasso_reg_model.fit(x_train, y_train)

# K-fold cross validation
# importing cross _val_ score function

lasso_scores = cross_val_score(lasso_reg_model, x_train, y_train, scoring='r2', cv=10)
print(lasso_scores)

# print average score of training data
print("Accuracy on Training data:", np.mean(lasso_scores))

# print the scores of test data and first work predict

lasso_test_data_predict = cross_val_predict(lasso_reg_model, x_test, y_test)
lasso_scores_test = cross_val_score(lasso_reg_model, x_test, y_test, cv=10)
print(lasso_scores_test)

print("Accuracy on Testing data:", np.mean(lasso_scores_test))

# R square error
lasso_error_score = metrics.r2_score(y_test, lasso_test_data_predict)
print("R squared error:", lasso_error_score)
print("Mean absolute error:", metrics.mean_absolute_error(y_test, lasso_test_data_predict))

# visualize the actual prices and predicted prices

plt.scatter(x=y_test, y=lasso_test_data_predict, c=y_test, cmap='Paired_r', edgecolors='black')
plt.colorbar()
plt.ylabel("Actual Price")
plt.xlabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


# "Model training using Random Forest"
print("....RandomForest Model....")
# loading RandomForest model
rf_reg_model = RandomForestRegressor()
rf_reg_model.fit(x_train, y_train)

# K-fold cross validation
# importing cross _val_ score function

rf_reg_scores = cross_val_score(rf_reg_model, x_train, y_train, scoring='r2', cv=10)
print(rf_reg_scores)

# print average score of training data
print("Accuracy on Training data:", np.mean(rf_reg_scores))

# print the scores of test data and first work predict

rf_reg_test_data_predict = cross_val_predict(rf_reg_model, x_test, y_test)
rf_reg_scores_test = cross_val_score(rf_reg_model, x_test, y_test, cv=10)
print(rf_reg_scores_test)

print("Accuracy on Testing data:", np.mean(rf_reg_scores_test))

# R square error
rf_reg_error_score = metrics.r2_score(y_test, rf_reg_test_data_predict)
print("R squared error:", rf_reg_error_score)
print("Mean absolute error:", metrics.mean_absolute_error(y_test, rf_reg_test_data_predict))

# visualize the actual prices and predicted prices

plt.scatter(x=y_test, y=rf_reg_test_data_predict, c=y_test, cmap='Paired_r', edgecolors='black')
plt.colorbar()
plt.ylabel("Actual Price")
plt.xlabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


# # Loading Model into file Using Pickle
# with open('RF_price_predict_model.pkl', 'wb') as file:
#     pickle.dump(rf_reg_model, file)











