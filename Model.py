import numpy as np
import matplotlib as plt
import pandas as pd

#Loading a file

# Adjust display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows (use with caution for large DataFrames)
pd.set_option('display.width', None)        # Adjust console width to fit all columns
pd.set_option('display.max_colwidth', None) # Show full content of each column

victimBasedCrime_df = pd.read_csv("D:\\IIT\\2nd Year\\Group project\\daatasets\\Usable\\criminal_profiling_data_fixed.csv")

# #Preprcess has started
# victimBasedCrime_df = victimBasedCrime_df.drop(columns='Name')
#
# victimBasedCrime_df[['Age','sex']] = victimBasedCrime_df['Victim Info'].str.split(',',expand=True)
#
# victimBasedCrime_df = victimBasedCrime_df.drop(columns='Victim Info')



print(victimBasedCrime_df.head())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Step 1: Extract 'Age' from 'Date of Birth' if it’s not correct
df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')
df['Age'] = pd.to_datetime('today').year - df['Date of Birth'].dt.year  # Calculate age from DOB
df.drop(columns=['Date of Birth'], inplace=True)  # Drop 'Date of Birth' column

# Step 2: Handle missing values
df['Gender'].fillna('Unknown', inplace=True)  # Fill missing gender with 'Unknown' or mode
df['Residence'].fillna('Unknown', inplace=True)  # Similarly for Residence
df['Crime Date'].fillna('Unknown', inplace=True)  # Optional: Drop or fill missing crime date
df['Crime Time'].fillna('Unknown', inplace=True)  # Optional: Drop or fill missing crime time
df['Evidence Collected'].fillna('Unknown', inplace=True)  # Fill missing values with 'Unknown'
df['Latitude'].fillna(df['Latitude'].median(), inplace=True)  # Fill missing Latitudes with median
df['Longitude'].fillna(df['Longitude'].median(), inplace=True)  # Fill missing Longitudes with median

# Step 3: Convert categorical features to numerical (Label Encoding or One-Hot Encoding)
label_encoder = LabelEncoder()

# Encoding 'Gender', 'Nationality', and 'Criminal Record'
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Nationality'] = label_encoder.fit_transform(df['Nationality'])
df['Criminal Record'] = label_encoder.fit_transform(df['Criminal Record'])

# For 'Crime Type', we'll use One-Hot Encoding as it could have many categories
df = pd.get_dummies(df, columns=['Crime Type'], drop_first=True)

# Step 4: Extract features from 'Crime Date' and 'Crime Time'
df['Crime Date'] = pd.to_datetime(df['Crime Date'], errors='coerce')
df['Crime Time'] = pd.to_datetime(df['Crime Time'], errors='coerce')

# Extract day of the week, month, and year from 'Crime Date'
df['Crime Day'] = df['Crime Date'].dt.dayofweek  # Monday = 0, Sunday = 6
df['Crime Month'] = df['Crime Date'].dt.month
df['Crime Year'] = df['Crime Date'].dt.year

# Extract hour and minute from 'Crime Time'
df['Crime Hour'] = df['Crime Time'].dt.hour
df['Crime Minute'] = df['Crime Time'].dt.minute

# Drop the original 'Crime Date' and 'Crime Time' columns as we have extracted features
df.drop(columns=['Crime Date', 'Crime Time'], inplace=True)

# Step 5: Drop irrelevant columns (e.g., 'Name', 'Victim Info')
df.drop(columns=['Name', 'Victim Info'], inplace=True)

# Step 6: Separate features (X) and target (y)
X = df.drop(columns=['Age'])  # 'Age' is the target variable
y = df['Age']  # 'Age' is the target variable

# Step 7: Scale numerical features (optional, but recommended)
scaler = StandardScaler()
X[['Latitude', 'Longitude']] = scaler.fit_transform(X[['Latitude', 'Longitude']])

# Step 8: Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 9: Initialize and train the Gradient Boosting Regressor model
regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
regressor.fit(X_train, y_train)

# Step 10: Make predictions on the test set
y_pred = regressor.predict(X_test)

# Step 11: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Optionally, print some predictions vs. actual values for comparison
print("Predictions:", y_pred[:10])  # First 10 predictions
print("Actual:", y_test[:10])  # First 10 actual values
