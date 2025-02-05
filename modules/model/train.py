from modules.data import load
from modules.model.baseline import baseline_model
import tensorflow as tf

def train_model(export='keras', num_filters=32, dropout_rate=0.2, lr=1e-4):
    (X_train, y_train), (X_val, y_val), _ = load.get_dataset()
    model = baseline_model(
        filters=num_filters,
        dropout=dropout_rate,
        lr=lr
    )

    model.fit(X_train, y_train, epochs=CONFIG['optimize']['epochs'],
              batch_size=CONFIG['optimize']['batch_size'],
              validation_data=(X_val, y_val))
    model.save(CONFIG['path']['model']['keras'])

    model_keras = tf.keras.models.load_model(CONFIG['path']['model']['keras'])
    model_keras.save(CONFIG['path']['model']['h5'], save_format="h5")
