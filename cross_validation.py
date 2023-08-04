from os import listdir
from os.path import join
import torch
from constants import RESISC45_DATASET_FILE, UCMERCEDLU_DATASET_FILE
import pandas as pd
from utils import get_img_trainsforms_val_v1, DatasetConfig, DatasetTypeEnum, DatasetClassification, DataLoader, evaluate_classification_model, get_device


def get_models(models):
    for model in models:
        val_fold = int(model.split('_')[-1][:-3])
        print(val_fold)
        dataset = None
        if RESISC45_DATASET_FILE in model:
            dataset = RESISC45_DATASET_FILE
        if UCMERCEDLU_DATASET_FILE in model:
            dataset = UCMERCEDLU_DATASET_FILE

        val_config = DatasetConfig(
            dataset_file=dataset,
            transform=get_img_trainsforms_val_v1(),
            dataset_type=DatasetTypeEnum.val,
            val_fold=val_fold,
        )
        val_set = DatasetClassification(val_config)
        val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False)

        yield (model, torch.load(model), val_fold, dataset, val_dataloader)


def filelist_models(input_path):
    models = listdir(input_path)
    models = [
        join(input_path, model) for model in models if model[-3:] == '.pt'
    ]
    return models


def cross_validation_by_dataset(input_path):
    models = filelist_models(input_path)
    results = {}

    for name, model, val_fold, dataset, val_dataloader in get_models(models):
        result = evaluate_classification_model(
            val_dataloader,
            model,
            get_device()
        )

        print(result.accuracy)

        if dataset not in results:
            results[dataset] = {}

        results[dataset][val_fold] = result

    # aggregate results
    avg_results = {}
    for dataset, folds in results.items():
        mean_acc = 0
        count = 0
        for val_fold, result in folds.items():
            mean_acc += result.accuracy
            count += 1
        mean_acc /= count
        avg_results[dataset] = mean_acc

    print(avg_results)


if __name__ == "__main__":
    cross_validation_by_dataset("/Users/cristianion/Desktop/satimg_model/cls_models")
