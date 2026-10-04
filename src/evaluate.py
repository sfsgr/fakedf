import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    test_data = np.load(os.path.join("data", "processed", "test.npz"))
    x_test, y_test = test_data["image"], test_data["label"]
    
    model = tf.keras.models.load_model(os.path.join("models", "model.h5"))
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()
