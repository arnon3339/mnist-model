from modules.data import load
from modules.model.baseline import baseline_model
import tensorflow as tf
import tf2onnx

def train_model(export='keras', num_filters=32, dropout_rate=0.2, lr=1e-4):
    (X_train, y_train), (X_val, y_val), _ = load.get_dataset()
    model = baseline_model(
        filters=num_filters,
        dropout=dropout_rate,
        lr=lr
    )

    model.fit(X_train, y_train, epochs=CONFIG['tuned']['epochs'],
              batch_size=CONFIG['tuned']['batch_size'],
              validation_data=(X_val, y_val))
    model.save(CONFIG['path']['model']['keras'])

    model_keras = tf.keras.models.load_model(CONFIG['path']['model']['keras'])
    model_keras.save(CONFIG['path']['model']['h5'], save_format="h5")

    if export == "onnx":
        keras2onnx()

def keras2tflite():
    model = tf.keras.models.load_model(CONFIG['path']['model']['keras'])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(CONFIG['path']['model']['tflite'], "wb") as f:
        f.write(tflite_model)

def keras2onnx():
    model = tf.keras.models.load_model(CONFIG['path']['model']['keras'])

    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    # Save ONNX model
    with open(CONFIG['path']['model']['onnx'], "wb") as f:
        f.write(model_proto.SerializeToString())