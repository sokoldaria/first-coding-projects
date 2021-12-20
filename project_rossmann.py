# -*- coding: utf-8 -*-

# PROJECT ML: Prediction of Store Sales with a DNN Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data

df = pd.read_csv("rossmannsales.csv")
print(df.head())
print(df.shape)

store = pd.read_csv("rossmannpromotion.csv")
print(store.head())
print(store.shape)


# Merge the data

data = df.merge(store, on = ["Store"], how = "inner")
print(data.head())
print(data.shape)
print(data.dtypes)

print('Distinct number of Stores:', len(data["Store"].unique()))
print('Distinct number of Dates:', len(data["Date"].unique()))
print('Average daily sales of all Stores:', round(data["Sales"].mean(), 2))
print(data["DayOfWeek"].value_counts())


# Create new columns related to dates

data["Date"] = pd.to_datetime(data["Date"], infer_datetime_format = True)

data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day

print(data.iloc[:,-3:].head())


# Visualize the data

data.hist(figsize=(20,12), color="#107009AA")
plt.savefig("data.png", dpi=100)
plt.show()

plt.figure(figsize=(20,12))
sns.heatmap(data.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
plt.savefig("corr.png", dpi=100)
plt.show()


# Handle the missing values

print(data.isnull().sum())
data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].mean())


# Encode the data

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

target = ["Sales"]
numeric_col = ["Customers", "Open", "Promo", "Promo2", "StateHoliday", "SchoolHoliday", "CompetitionDistance"]
categorical_col = ["DayOfWeek", "Month", "Year", "StoreType", "Assortment"]

def create_encode(df, col):
    le = LabelEncoder()
    a = le.fit_transform(data[col]).reshape(-1,1)
    ohe = OneHotEncoder(sparse=False)
    col_names = [col+ "_" + str(i) for i in le.classes_]
    print(col_names)
    return (pd.DataFrame(ohe.fit_transform(a), columns = col_names))

temp = data[numeric_col]

for col in categorical_col:
    temp_df = create_encode(data, col)
    temp = pd.concat([temp, temp_df], axis=1)
    
print("Shape of Data: ", temp.shape)
print("Distinct Datatypes: ", temp.dtypes.unique())

temp["StateHoliday"] = np.where(temp["StateHoliday"]=="0", 0,1)
temp.dtypes.unique()


# Split the data

from sklearn.model_selection import train_test_split

X = temp
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_val: ", X_val.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_val: ", y_val.shape)
print("Shape of y_test: ", y_test.shape)


# Standardize the data

# from sklearn.preprocessing import MinMaxScaler  
# scaler = MinMaxScaler()
# scaler.fit(X_train)  
# X_train_scaled = scaler.transform(X_train)  
# X_test_scaled = scaler.transform(X_test)  
# X_val_scaled = scaler.transform(X_val)  


# Create a DNN model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(72, input_dim=36, activation="relu"))
model.add(Dense(72, activation="relu"))
model.add(Dense(72, activation="relu"))
model.add(Dense(1, activation="linear"))

model.summary()

from contextlib import redirect_stdout

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])


# Train the model

from tensorflow.keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor="mean_absolute_error", patience=3)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, callbacks=[callback], batch_size=128)


# Evaluate the model

prediction = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print(mean_squared_error(y_test, prediction))
print(mean_absolute_error(y_test, prediction))
print(r2_score(y_test, prediction))
