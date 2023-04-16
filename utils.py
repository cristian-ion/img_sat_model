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
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
import enum
from pydantic import BaseModel
import torch.optim as optim
import platform


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


class DatasetTypes(enum.Enum):
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


class PretrainedModels(enum.Enum):
    resnet = "resnet"
    alexnet = "alexnet"
    vgg = "vgg"
    squeezenet = "squeezenet"
    densenet = "densenet"
    inception = "inception"


class PretrainedModelConfig(BaseModel):
    model_name: PretrainedModels
    num_classes: int
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract: bool
    use_pretrained: bool = True


def initialize_pretrained_model(pretrained_model_config: PretrainedModelConfig):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_name = pretrained_model_config.model_name
    num_classes = pretrained_model_config.num_classes
    feature_extract = pretrained_model_config.feature_extract
    use_pretrained = pretrained_model_config.use_pretrained

    model_ft = None
    input_size = 0

    if model_name == PretrainedModels.resnet:
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == PretrainedModels.alexnet:
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == PretrainedModels.vgg:
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == PretrainedModels.squeezenet:
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == PretrainedModels.densenet:
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == PretrainedModels.inception:
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


def get_pretrained_transforms(input_size: int, dataset_type: DatasetTypes):
    """
    Get transforms for train

    input_size: 224, 256, etc.
    """
    if dataset_type == DatasetTypes.train:
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif dataset_type == DatasetTypes.val:
        data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        raise Exception(f"Dataset type not known {str(dataset_type)}")
    return data_transforms


def get_pretrained_optimizer(model_ft, feature_extract=True):
    params_to_update = get_params_requires_grad(model_ft, feature_extract=feature_extract)
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer_ft


class DatasetConfig(BaseModel):
    dataset_file: str
    dataset_type: Optional[DatasetTypes] = None
    val_fold: Optional[int] = None
    shuffle: bool = False
    transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    resize_res_x: Optional[int] = None
    resize_res_y: Optional[int] = None


class DatasetResisc45(Dataset):
    def __init__(self, dataset_config: DatasetConfig):
        val_fold = dataset_config.val_fold or None
        df = pd.read_csv(dataset_config.dataset_file)
        if dataset_config.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        folds = list(df["fold"].unique())  # get all folds
        if val_fold:
            if val_fold not in folds:
                raise Exception("Fold not found.")
            if dataset_config.dataset_type == DatasetTypes.train:
                df.drop(index=df[df["fold"] == val_fold].index, inplace=True)  # drop the validation fold
            elif dataset_config.dataset_type == DatasetTypes.val:
                df = df[df["fold"] == val_fold]  # keep only the validation fold in dataset
        self.img_labels = df
        self.transform = dataset_config.transform
        self.target_transform = dataset_config.target_transform
        self.labels = list(df['label'].unique())
        self.num_classes = len(self.labels)
        self.resy = dataset_config.resize_res_y
        self.resx = dataset_config.resize_res_x

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label_index = self.img_labels.iloc[idx, 2]
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


def plot_validation_epochs(epochs, train_loss, val_loss, fold):
    """
    Show plot train and validation by epoch.
    """
    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label='train loss')
    ax.plot(epochs, val_loss, label='val loss')
    ax.legend()
    #specify axis tick step sizes
    _ = plt.xticks(np.arange(min(epochs), max(epochs)+1, 1))
    # _ = plt.yticks(np.arange(0, max(y)+0.1, 0.1))

    ax.set_title(f"Train validation by epoch. Fold {fold}")
    plt.show()



def train_one_epoch(loss_fn, optimizer, dataloader, model, device: str):
    """
    Train for one epoch.

    loss_fn: CrossEntropy, etc.
    optimizer: SGD, etc.
    dataloader: DataLOader
    model: neural network
    device: cpu, gpu, mps
    """
    num_batches = len(dataloader)

    print("Started train.")
    size = len(dataloader.dataset)
    
    model.train()  # set model to train mode

    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

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


def val_one_epoch(loss_fn, dataloader, model, device):
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
            test_loss += loss_fn(pred, y).item()
            # compare pred with y
            pred_vs_y = (nn.Softmax(dim=1)(pred).argmax(1) == y.argmax(1)).type(torch.float)
            correct += pred_vs_y.sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


class TrainConfig(BaseModel):
    num_epochs: int
    batch_size: int
    val_fold: int


def train_pretrained_resnet(train_config: TrainConfig):
    """
    Train pretrained resnet, with feature extract
    """
    print_versions()
    device = get_device()

    df = pd.read_csv("dataset_resisc45.csv")
    num_classes = len(list(df['label'].unique()))  # take number of classes from datatset

    pretrained_model_config = PretrainedModelConfig(
        model_name=PretrainedModels.resnet,
        num_classes=num_classes,
        feature_extract=True,
        use_pretrained=True,
    )
    
    model_ft, input_size = initialize_pretrained_model(pretrained_model_config)
    print(model_ft)
    model_ft.to(device)

    optimizer_ft = get_pretrained_optimizer(model_ft, pretrained_model_config.feature_extract)
    print(optimizer_ft)


    print(f"Fold: {train_config.val_fold}\n------------------------------")

    # train config
    dataset_train_config = DatasetConfig(
        dataset_file="dataset_resisc45.csv",
        transform=get_pretrained_transforms(input_size, DatasetTypes.train),
        dataset_type=DatasetTypes.train,
        val_fold=train_config.val_fold,
    )
    # validation config
    dataset_val_config = DatasetConfig(
        dataset_file="dataset_resisc45.csv",
        transform=get_pretrained_transforms(input_size, DatasetTypes.val),
        dataset_type=DatasetTypes.val,
        val_fold=train_config.val_fold,
    )

    # create train and val set
    train_set = DatasetResisc45(dataset_train_config)
    val_set = DatasetResisc45(dataset_val_config)
    train_dataloader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=train_config.batch_size,
        shuffle=False
    )

    # loss
    loss_fn = nn.CrossEntropyLoss()
    # optimizer
    optimizer_ft = get_pretrained_optimizer(model_ft, pretrained_model_config.feature_extract)

    # Train and evaluate
    train_losses = []
    val_losses = []
    for t in range(train_config.num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_one_epoch(loss_fn, optimizer_ft, train_dataloader, model_ft, device)
        val_loss = val_one_epoch(loss_fn, val_dataloader, model_ft, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    plot_validation_epochs([i for i in range(train_config.num_epochs)], train_losses, val_losses, train_config.val_fold)


if __name__ == "__main__":
    # test utils
    print_versions()

    # show plot train and validation by epoch
    plot_validation_epochs(np.arange(0, 5, 1), np.exp([5, 4, 3, 2, 1]), np.exp([7, 6, 5, 4, 3]), 1)
    dataset_config = DatasetConfig(
        dataset_file="dataset_resisc45.csv",
        resize_res_x=256,
        resize_res_y=256,
    )

    train_set = DatasetResisc45(dataset_config)
    val_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    # print first sample
    print(train_set[0])
