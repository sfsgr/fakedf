import os
import yaml
import numpy as np
import csv
import tensorflow as tf
def main():
    os.makedirs("models", exist_ok=True)
    with open("params.yaml", "r") as f:
        params =yaml.safe_load(f)["train"]
    train_data=np.load(os.path.join("data","processed","train.npz"))
    val_data=np.load(os.path.join("data","processed","val.npz"))
    x_train,y_train=train_data["image"],train_data["label"]
    x_val,y_val=val_data["image"],val_data["label"]
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(params["dense_units"], activation='relu'),
        tf.keras.layers.Dropout(params["dropout_rate"]),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    history=model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=params["epochs"],batch_size=params["batch_size"])
    model.save(os.path.join("models","model.h5"))
    keys=history.history.keys()
    with open(os.path.join("models","history.csv"),"w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[history.history[k] for k in keys]))
if __name__ =="__main__":
    main()
