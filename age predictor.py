import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # For imputing missing values

# Load your dataset
df = pd.read_csv("D:\\IIT\\2nd Year\\Group project\\daatasets\\Usable\\criminal_profiling_data_fixed.csv")

#Extract 'Age' from 'Date of Birth' if it’s not correct
df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')
df['Age'] = pd.to_datetime('today').year - df['Date of Birth'].dt.year  # Calculate age from DOB
df.drop(columns=['Date of Birth'], inplace=True)  # Drop 'Date of Birth' column
df.drop(columns='Name', inplace=True)  # Drop the 'Name' column
print("*********************")
print(df['Age'])

#Split 'Victim Info' into two new columns: 'victim_age' and 'victim_sex'
# Extract 'victim_sex' and 'victim_age' from 'Victim Info' column
df[['victim_sex', 'victim_age']] = df['Victim Info'].str.split(',', expand=True)
df['victim_age'] = df['victim_age'].str.extract(r'(\d+)').astype(float)  # Use raw string for regex
df['victim_sex'] = df['victim_sex'].map({'Male': 0, 'Female': 1})  # Convert sex to binary (0 = Male, 1 = Female)
df.drop(columns=['Victim Info'], inplace=True)  # Drop the original 'Victim Info' column

#  Handle missing values (fill with mode or appropriate value)
# Use SimpleImputer to handle missing values
imputer = SimpleImputer(strategy='most_frequent')  # Impute categorical columns with most frequent value
df[['Gender', 'Residence', 'Crime Date', 'Crime Time', 'Evidence Collected']] = imputer.fit_transform(df[['Gender', 'Residence', 'Crime Date', 'Crime Time', 'Evidence Collected']])

# For numerical columns, use median imputation
numerical_imputer = SimpleImputer(strategy='median')
df[['Latitude', 'Longitude', 'victim_age']] = numerical_imputer.fit_transform(df[['Latitude', 'Longitude', 'victim_age']])

#  Encode categorical features using LabelEncoder or One-Hot Encoding

# Label Encoding for columns with a small set of categories
label_encoder = LabelEncoder()

# Encoding 'Gender', 'Nationality', 'Criminal Record', and 'Residence'
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Nationality'] = label_encoder.fit_transform(df['Nationality'])
df['Criminal Record'] = label_encoder.fit_transform(df['Criminal Record'])
df['Residence'] = label_encoder.fit_transform(df['Residence'])

# For 'Crime Type' and 'Evidence Collected', we will use One-Hot Encoding
df = pd.get_dummies(df, columns=['Crime Type', 'Evidence Collected'], drop_first=True)

# Extract features from 'Crime Date' and 'Crime Time'
df['Crime Date'] = pd.to_datetime(df['Crime Date'], errors='coerce')

# Specify the format for 'Crime Time' to avoid parsing issues
df['Crime Time'] = pd.to_datetime(df['Crime Time'], format='%H:%M:%S', errors='coerce')  # If time is HH:MM:SS format

# Extract day of the week, month, and year from 'Crime Date'
df['Crime Day'] = df['Crime Date'].dt.dayofweek  # Monday = 0, Sunday = 6
df['Crime Month'] = df['Crime Date'].dt.month
df['Crime Year'] = df['Crime Date'].dt.year

# Extract hour and minute from 'Crime Time'
df['Crime Hour'] = df['Crime Time'].dt.hour
df['Crime Minute'] = df['Crime Time'].dt.minute

# Drop the original 'Crime Date' and 'Crime Time' columns as we have extracted features
df.drop(columns=['Crime Date', 'Crime Time'], inplace=True)

# Check for missing values (NaN)
print("Missing values in each column:\n", df.isna().sum())  # Check for missing values in each column

# Handle any remaining missing values (if any)
# Use SimpleImputer to fill any remaining NaN values
final_imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(final_imputer.fit_transform(df), columns=df.columns)

# Check if all NaNs are handled
print("Missing values after imputation:\n", df.isna().sum())

# Separate features (X) and target (y)
X = df.drop(columns=['Age'])  # 'Age' is the target variable
y = df['Age']  # 'Age' is the target variable

# Scale numerical features (optional, but recommended)dbdbdbfd
scaler = StandardScaler()
X[['Latitude', 'Longitude']] = scaler.fit_transform(X[['Latitude', 'Longitude']])

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Gradient Boosting Regressor model
regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Optionally, print some predictions vs. actual values for comparison
print("Predictions:", y_pred[:10])  # First 10 predictions
print("Actual:", y_test[:10])  # First 10 actual values
