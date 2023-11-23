import pandas as pd


def plot_train_val(fname):
    df = pd.read_csv(fname, sep='\t')
    print(df.head())
    df.plot(x='epoch', y=['train_loss', 'val_loss'], figsize=(10, 10))
    df.plot(x='epoch', y=['train_error_rate', 'val_error_rate'], figsize=(10, 10))


plot_train_val("/Users/cristianion/Desktop/img_sat_model/models/inria/inria_model_1_0_7_val.tsv")