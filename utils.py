import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import enum
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from pydantic import BaseModel
from typing import Optional, Any
from torchvision.transforms import Compose
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import enum
from pydantic import BaseModel
import torch.optim as optim
import platform
from torchvision.transforms.functional import InterpolationMode


def print_versions():
    print(platform.platform())  # print current platform
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)


class Device(enum.StrEnum):
    cuda = 'cuda'
    mps = 'mps'
    cpu = 'cpu'


def get_device():
    # find CUDA / MPS / CPU device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


class DatasetTypeEnum(enum.Enum):
    train = "train"
    val = "val"
    test = "test"


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_params_requires_grad(model_ft, feature_extract):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    return params_to_update


class PretrainedModelsEnum(enum.Enum):
    resnet18 = "resnet18"
    alexnet = "alexnet"
    vgg = "vgg"
    squeezenet = "squeezenet"
    densenet = "densenet"
    inception = "inception"


class CNNParams(BaseModel):
    dataset_file: str
    model_name: PretrainedModelsEnum
    num_classes: int
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract: bool
    batch_size: int
    num_epochs: int
    criterion_name: str
    optimizer_name: str
    train_transforms: Any
    val_transforms: Any
    use_pretrained: bool = True
    lr: float
    momentum: float


def get_pretrained_model(pretrained_model_config: CNNParams):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_name = pretrained_model_config.model_name
    num_classes = pretrained_model_config.num_classes
    feature_extract = pretrained_model_config.feature_extract
    use_pretrained = pretrained_model_config.use_pretrained

    model_ft = None
    input_size = 0

    if model_name == PretrainedModelsEnum.resnet18:
        """ Resnet18 with IMAGENET1K_V1 weights
        """
        # model_ft = models.resnet18(pretrained=use_pretrained)
        model_ft = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == PretrainedModelsEnum.alexnet:
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == PretrainedModelsEnum.vgg:
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == PretrainedModelsEnum.squeezenet:
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == PretrainedModelsEnum.densenet:
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == PretrainedModelsEnum.inception:
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    else:
        raise Exception("Invalid model name, exiting...")

    return model_ft, input_size


def get_img_transforms_train_v1():
    img_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return img_transforms


def get_img_trainsforms_val_v1():
    img_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return img_transforms


def get_pretrained_transforms(input_size: int, dataset_type: DatasetTypeEnum):
    """
    Get transforms for train

    input_size: 224, 256, etc.
    """
    if dataset_type == DatasetTypeEnum.train:
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif dataset_type == DatasetTypeEnum.val:
        data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        raise Exception(f"Dataset type not known {str(dataset_type)}")
    return data_transforms


class DatasetConfig(BaseModel):
    dataset_file: str
    dataset_type: Optional[DatasetTypeEnum] = None
    val_fold: Optional[int] = None
    shuffle: bool = False
    transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    resize_res_x: Optional[int] = None
    resize_res_y: Optional[int] = None


class DatasetClassification(Dataset):
    def __init__(self, dataset_config: DatasetConfig):
        val_fold = dataset_config.val_fold
        df = pd.read_csv(dataset_config.dataset_file)
        
        if dataset_config.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        folds = list(df["fold"].unique())  # get all folds
        if val_fold is not None:
            if val_fold not in folds:
                raise Exception("Fold not found.")
            if dataset_config.dataset_type == DatasetTypeEnum.train:
                df = df[df["fold"] != val_fold]  # keep train folds
            elif dataset_config.dataset_type == DatasetTypeEnum.val:
                df = df[df["fold"] == val_fold]  # keep validation folds

        self.df = df
        self.transform = dataset_config.transform
        self.target_transform = dataset_config.target_transform
        self.labels = list(df['label'].unique())
        self.num_classes = len(self.labels)
        self.resy = dataset_config.resize_res_y
        self.resx = dataset_config.resize_res_x

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label_index = self.df.iloc[idx, 2]
        # convert labels to one-hot.
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[label_index] = 1.0

        if self.transform:
            # pytorch transforms
            image = Image.open(img_path)
            image = self.transform(image)
        else:
            # opencv transforms
            image = cv2.imread(img_path)
            image = cv2.resize(image, dsize=(self.resy, self.resx))
            image = image - 127.5
            image = np.moveaxis(image, -1, 0).astype(np.float32) / 255.0  # move channels first (3 x RES_Y x RES_X)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def plot_validation_epochs(num_epochs, train_loss, val_loss, fold):
    """
    Show plot train and validation by epoch.
    """
    epochs = np.arange(0, num_epochs, 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label='train loss')
    ax.plot(epochs, val_loss, label='val loss')
    ax.legend()
    #specify axis tick step sizes
    _ = plt.xticks(np.arange(min(epochs), max(epochs)+1, 1))
    # _ = plt.yticks(np.arange(0, max(y)+0.1, 0.1))

    ax.set_title(f"Train validation by epoch. Fold {fold}")
    plt.show()



def train_one_epoch(criterion, optimizer, dataloader, model, device: str):
    """
    Train for one epoch.

    criterion: CrossEntropy, etc.
    optimizer: SGD, etc.
    dataloader: DataLOader
    model: neural network
    device: cpu, gpu, mps
    """
    num_batches = len(dataloader)

    print("Started train.")

    print(device)

    size = len(dataloader.dataset)
    
    model.train()  # set model to train mode

    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    return train_loss


def val_one_epoch(criterion, dataloader, model, device):
    """
    Validation for one epoch.

    loss_fn: CrossEntropy, etc.
    dataloader: DataLOader
    model: neural netowork
    device: cpu, gpu, mps
    """
    print("Started validation.")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.eval() # set model to evaluation mode

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            # compare pred with y
            pred_vs_y = (nn.Softmax(dim=1)(pred).argmax(1) == y.argmax(1)).type(torch.float)
            correct += pred_vs_y.sum().item()
    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # todo : also return f1, precision, recall, confusion matrix in one object;
    return test_loss


class EvaluateResult(BaseModel):
    accuracy: float
    mse: float


def squared_error(preds, target):
    return ((preds - target)**2).sum().item()


def evaluate_classification_model(dataloader, model, device) -> EvaluateResult:
    print("Started evaluation")
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    df = dataloader.dataset.df

    print(df.head())

    model.eval()

    correct = 0
    mse = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            pred_softmax = nn.Softmax(dim=1)(logits)

            pred_vs_y = (pred_softmax.argmax(1) == y.argmax(1)).type(torch.float)
            correct += pred_vs_y.sum().item()
            mse += squared_error(pred_softmax, y) / len(logits)

    correct /= dataset_size
    mse /= num_batches

    print(f"Accuracy: {correct}")
    print(f"MSE: {mse}")

    # todo: 
    # Positives, Negatives, FP, TP, FN, TN by class
    # roc curve
    # move evaluation in another script :) with classifier file as parameter and dataset file.
    # this script will do only evaluation and plots for classification.

    return EvaluateResult(accuracy=correct, mse=mse)


def get_cross_entropy_loss():
    return nn.CrossEntropyLoss()


def get_sgd_optimizer(model, feature_extract, lr=0.001, momentum=0.9):
    params_to_update = get_params_requires_grad(model, feature_extract)
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
    return optimizer


def resnet_pretrained_cross_validation(dataset_file, num_epochs, batch_size, num_folds=5):
    """
    Train pretrained resnet, with feature extract cross validation
    """
    print_versions()
    device = get_device()

    df = pd.read_csv(dataset_file)
    num_classes = len(list(df['label'].unique()))  # take number of classes from datatset

    pretrained_model_config = CNNParams(
        model_name=PretrainedModelsEnum.resnet,
        num_classes=num_classes,
        feature_extract=True,
        use_pretrained=True,
    )

    clf, input_size = get_pretrained_model(pretrained_model_config)
    print(clf)
    clf.to(device)


    dataset_train_config = DatasetConfig(
        dataset_file="dataset_resisc45.csv",
        transform=get_pretrained_transforms(input_size, DatasetTypeEnum.train),
        dataset_type=DatasetTypeEnum.train,
        val_fold=val_fold,
    )
    dataset_val_config = DatasetConfig(
        dataset_file="dataset_resisc45.csv",
        transform=get_pretrained_transforms(input_size, DatasetTypeEnum.val),
        dataset_type=DatasetTypeEnum.val,
        val_fold=val_fold,
    )

    train_set = DatasetClassification(dataset_train_config)
    val_set = DatasetClassification(dataset_val_config)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    fold_stats = []
    fold_train_losses = []
    fold_val_losses = []

    for val_fold in range(num_folds):
        print(f"Fold: {val_fold}\n------------------------------")

        criterion = nn.CrossEntropyLoss()
        params_to_update = get_params_requires_grad(clf, pretrained_model_config.feature_extract)
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        print(optimizer)

        train_losses = []
        val_losses = []  
        for t in range(num_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train_one_epoch(criterion, optimizer, train_dataloader, clf, device)
            val_loss = val_one_epoch(criterion, val_dataloader, clf, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        eval_result = evaluate_classification_model(val_dataloader, clf, device)
        
        fold_stats.append(eval_result)
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
    
    avg_accuracy = np.mean([stat.accuracy for stat in fold_stats])
    avg_mse = np.mean([stat.mse for stat in fold_stats])





if __name__ == "__main__":
    # test utils
    # print_versions()

    # # show plot train and validation by epoch
    # plot_validation_epochs(np.arange(0, 5, 1), np.exp([5, 4, 3, 2, 1]), np.exp([7, 6, 5, 4, 3]), 1)
    # dataset_config = DatasetConfig(
    #     dataset_file="dataset_resisc45.csv",
    #     resize_res_x=256,
    #     resize_res_y=256,
    # )

    # train_set = DatasetResisc45(dataset_config)
    # val_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    # # print first sample
    # print(train_set[0])

    print("Start.")

    results = resnet_pretrained_cross_validation(
        dataset_file="dataset_resisc45.csv",
        num_epochs=0,
        batch_size=32,
    )
    # plot_validation_epochs(15, results.train_losses, results.val_losses, fold=val_fold)

    print("Done.")
