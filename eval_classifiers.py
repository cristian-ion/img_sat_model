import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import DatasetResisc45, DatasetConfig, get_pretrained_transforms, DatasetTypes, initialize_pretrained_model, PretrainedModelConfig



def cross_validation():
    # load saved model from each fold and validate using validation set.
    dataset_file = "dataset_resisc45.csv"

    df = pd.read_csv(dataset_file)
    num_classes = len(list(df['label'].unique()))  # take number of classes from datatset

    print(num_classes)   

    for val_fold in range(5):
        model_file = f"model_fold{val_fold}"
        model.load_state_dict(torch.load(model_file))
        model.eval()
        # train and val datasets and loaders
        dataset_val_config = DatasetConfig(
            dataset_file=dataset_file,
            transform=get_pretrained_transforms(224, DatasetTypes.val),
            dataset_type=DatasetTypes.val,
            val_fold=val_fold,
        )
        val_set = DatasetResisc45(dataset_val_config)
        val_dataloader = DataLoader(
            val_set,
            batch_size=32,
            shuffle=False
        )


if __name__ == "__main__":
    print("Classification evaluation")