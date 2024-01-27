import os
import sys
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    train_output_file = sys.argv[1]
    val_output_file = sys.argv[2]

    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)

    X, y = datasets.load_iris(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=8888
    )

    np.savetxt(train_output_file, np.column_stack((X_train, y_train)), delimiter=",")
    np.savetxt(val_output_file, np.column_stack((X_val, y_val)), delimiter=",")
