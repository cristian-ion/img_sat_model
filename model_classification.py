import itertools

from utils import (
    DatasetConfig,
    DatasetTypeEnum,
    DatasetClassification,
    DataLoader,
    get_pretrained_model,
    get_cross_entropy_loss,
    get_sgd_optimizer,
    CNNParams,
    train_one_epoch,
    val_one_epoch,
)
from utils import *
from constants import RESISC45_DATASET_FILE, UCMERCEDLU_DATASET_FILE

import pandas as pd


MODELS_PATH = "./cls_models"
NUM_VAL_FOLDS = 5


def get_cnn_params_1() -> CNNParams:
    params = CNNParams(
        dataset_file=RESISC45_DATASET_FILE,
        model_name=PretrainedModelsEnum.resnet18,
        train_transforms=get_img_transforms_train_v1(),
        val_transforms=get_img_trainsforms_val_v1(),
        num_classes=45,
        batch_size=32,
        num_epochs=20,
        criterion_name="cross_entropy",
        optimizer_name="sgd",
        feature_extract=True,
        use_pretrained=True,
        lr=0.001,
        momentum=0.9,
    )
    return params


def get_cnn_params_3() -> CNNParams:
    params = CNNParams(
        dataset_file=UCMERCEDLU_DATASET_FILE,
        model_name=PretrainedModelsEnum.resnet18,
        train_transforms=get_img_transforms_train_v1(),
        val_transforms=get_img_trainsforms_val_v1(),
        num_classes=21,
        batch_size=32,
        num_epochs=20,
        criterion_name="cross_entropy",
        optimizer_name="sgd",
        feature_extract=True,
        use_pretrained=True,
        lr=0.001,
        momentum=0.9,
    )
    return params


def train_val_dataloaders(params: CNNParams, val_fold: int):
    train_config = DatasetConfig(
        dataset_file=params.dataset_file,
        transform=params.train_transforms,
        dataset_type=DatasetTypeEnum.train,
        val_fold=val_fold,
    )
    val_config = DatasetConfig(
        dataset_file=params.dataset_file,
        transform=params.val_transforms,
        dataset_type=DatasetTypeEnum.val,
        val_fold=val_fold,
    )

    train_set = DatasetClassification(train_config)
    val_set = DatasetClassification(val_config)

    train_dataloader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=params.batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def train_model_on_fold(params: CNNParams, val_fold: int, save_path: str):
    print(params.model_name, params.dataset_file, val_fold, save_path)

    with open(f"{MODELS_PATH}/model_{params.dataset_file}_{val_fold}", "w") as f:
        f.write(f"{str(params)}; {val_fold}; {save_path}")

    train_data, val_data = train_val_dataloaders(params, val_fold)

    cnn_model, input_size = get_pretrained_model(params)
    cnn_model.to(get_device())

    if params.criterion_name == "cross_entropy":
        criterion = get_cross_entropy_loss()
    if params.optimizer_name == 'sgd':
        optimizer = get_sgd_optimizer(cnn_model, feature_extract=params.feature_extract, lr=0.001, momentum=0.9)

    train_losses = []
    val_losses = []

    for epoch in range(params.num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        train_loss = train_one_epoch(criterion, optimizer, train_data, cnn_model, device=get_device())
        val_loss = val_one_epoch(criterion, val_data, cnn_model, device=get_device())
        
        # update train / val loss plot
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save(cnn_model, f"{MODELS_PATH}/model_{params.dataset_file}_{val_fold}.pt")
    
    pd.DataFrame({
        'train_losses': train_losses,
        'val_losses': val_losses,
    }).to_csv(f"{MODELS_PATH}/model_{params.dataset_file}_{val_fold}_loss.csv")

    print("done.")
    # save best torch model in save_path / dataset_file / model_name / val_fold


def train_models_cross_validation():
    # setup proposed models with different params
    cnn_models = [
        get_cnn_params_1(),
        get_cnn_params_3(),
    ]
    print(cnn_models)

    # train proposed models
    for params, val_fold in itertools.product(cnn_models, range(NUM_VAL_FOLDS)):
        train_model_on_fold(params, val_fold, save_path=MODELS_PATH)

    # cross validation

    # confusion matrix




if __name__ == "__main__":
    train_models_cross_validation()
