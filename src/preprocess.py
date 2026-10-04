import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
def main():
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    with open("params.yaml", "r") as f:
        params=yaml.safe_load(f)["preprocess"]
    train_data=np.load(os.path.join("data","raw","train.npz"))
    test_data=np.load(os.path.join("data","raw","test.npz"))
    x_train_full=(train_data["image"] / 255.0) * 0.9
    y_train_full=train_data["label"]
    x_test=test_data["image"] / 255.0
    y_test=test_data["label"]
    x_train,x_val,y_train,y_val=train_test_split(
        x_train_full,y_train_full,
        test_size=params["test_size"], 
        random_state=params["seed"]
    )
    np.savez_compressed(os.path.join("data","processed","train.npz"), image=x_train,label=y_train)
    np.savez_compressed(os.path.join("data","processed","val.npz"), image=x_val,label=y_val)
    np.savez_compressed(os.path.join("data","processed","test.npz"), image=x_test,label=y_test)
if __name__ == "__main__":
    main()
