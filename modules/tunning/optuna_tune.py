import optuna
import tensorflow as tf
from modules.data import load
from modules.model import baseline
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def objective(trial):
    (X_train, y_train), (X_val, y_val), _ = load.get_dataset()
    num_filters = trial.suggest_categorical("filters", CONFIG['tunning']['number_filters'])
    dropout_rate = trial.suggest_float("dropout",
                                        CONFIG['tunning']['dropout_rate'][0],
                                        CONFIG['tunning']['dropout_rate'][1]
                                       )
    learning_rate = trial.suggest_float("learning_rate",
                                                CONFIG['tunning']['learning_rate'][0],
                                                CONFIG['tunning']['learning_rate'][1],
                                                log=True
                                             )

    # model = Sequential([
    #     Conv2D(num_filters, (3,3), activation='relu', input_shape=(28,28,1)),
    #     MaxPooling2D((2,2)),
    #     Dropout(dropout_rate),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(dropout_rate),
    #     Dense(10, activation='softmax')
    # ])
    model = baseline.baseline_model(num_filters, dropout=dropout_rate, lr=learning_rate)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=CONFIG['tunning']['epochs'],
                        batch_size=CONFIG['tunning']['batch_size'], verbose=0,
                        validation_data=(X_val, y_val))

    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    return accuracy

def run(n_trials=10):
    # print(CONFIG)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials) 

    print("Best Hyperparameters:", study.best_params)
