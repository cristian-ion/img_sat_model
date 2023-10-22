from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    input_path = "/Users/cristianion/Desktop/img_sat_mdoel/cls_models"

    models = listdir(input_path)

    models = [
        join(input_path, model) for model in models if model.endswith("_loss.csv")
    ]

    for model in models:
        df = pd.read_csv(model, index_col=[0])

        print(df.head())

        train_losses = df["train_losses"]
        val_losses = df["val_losses"]
        epochs = range(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")

        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # Set the tick locations
        plt.xticks(np.arange(0, len(train_losses), 2))

        # Display the plot
        plt.legend(loc="best")

        plt.savefig(f"{model}.png")
