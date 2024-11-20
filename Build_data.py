import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_data():
    # Load the dataset
    file_path = "auto-mpg (1).csv"
    df = pd.read_csv(file_path)
    
    
    #skimming into the datset, I realised that the 'horsepower' coloumn has multiple missing data, so we need to fix it by either 
    #introducing a NaN value instead, or simply remove the row
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df = df.dropna() 
    
    # we decided to drop 'car name' feature, as it's not needed for training
    df = df.drop(columns=['car name'])
    
    
    X = df.drop(columns=['mpg'])
    y = df['mpg']
       
    # To further optimize the data, we have to do certain coloumn transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']),
            #normalize the numeric features dataset to have a variance of 0 and a mean of 1, as this helps us in training the model.
            #spefically, it helps us in the gradient decent, as the algorithm will be faster as the contours of the loss function will 
            #have a better shape that help the algorithm easily and accuratly locate the minima.
            ('cat', OneHotEncoder(), ['origin'])#as we are using a regression model with a ReLU activation function, we dont need the model to create any dependincies'
            #between the 3 numbers for which the origin is originally encoded, so we create three seperate coloumns for the origin.
        ])
    
    # Apply transformations to features
    X = preprocessor.fit_transform(X)
    
    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return  X_train, X_test, y_train, y_test
