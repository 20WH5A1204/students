import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

def read_data(file_path):
    return pd.read_csv(file_path)

def add_total_column(df):
    df['total'] = df['G1'] + df['G2'] + df['G3']
    return df

def select_columns(df, columns):
    return df[columns]

def encode_categorical(df, column_map):
    for col, mapping in column_map.items():
        df[col] = df[col].map(mapping)
    return df

def prepare_data(df, target_col, threshold):
    X = df.drop(columns=target_col)
    y = (df[target_col] >= threshold).astype(int)
    return X, y

def train_decision_tree(X_train, y_train):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def main():
    mat_df = read_data('C:\\Users\\Usha Sree\\OneDrive\\Documents\\Desktop\\students\\Dataset\\mat2.csv')
    mat_df = add_total_column(mat_df)
    
    cat_columns = ['school', 'studytime', 'health']
    matdf_selected = select_columns(mat_df, cat_columns)
    
    column_map = {'school': {'GP': 0, 'MS': 1}}
    matdf_selected = encode_categorical(matdf_selected, column_map)
    
    X, y = prepare_data(mat_df, 'total', 30)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    model = train_decision_tree(X_train, y_train)
    
    print("Accuracy:", model.score(X_test, y_test))
    
    # Example prediction
    example_data = [[0, 1, 3]]  # Example input data
    print("Predicted class for example data:", model.predict(example_data))

if __name__ == "__main__":
    main()
