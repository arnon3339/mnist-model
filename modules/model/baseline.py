from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import tensorflow as tf

def baseline_model(filters=32, dropout=0.2, lr=1e-4):
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(filters, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(dropout),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
