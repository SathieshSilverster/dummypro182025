
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_absolute_error, mean_squared_error  


df = pd.read_csv(r'D:\vscode\dummypro182025\messy_dataset_100k.csv')  
print(df.head())  # Display first 5 rows
print(df.info())  # Show column data types
print(df.describe())  # Summary statistics


df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean
print(df.isnull().sum())