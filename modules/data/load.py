import pandas as pd
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
import toml

def get_dataset():
    df = pd.read_csv(CONFIG['path']['dataset'], index_col=False)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0 
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    y_train, y_val, y_test = to_categorical(y_train, 10), to_categorical(y_val, 10),\
        to_categorical(y_test, 10)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)