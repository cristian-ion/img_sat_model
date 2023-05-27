import yaml
import torch
import torchvision
import torch.utils.data
from torchvision import transforms
import os.path
import pandas as pd

from imgcls.datasets import ImgClsDataset


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


def get_params_requires_grad(torch_model):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = torch_model.parameters()
    
    print("Params to learn:")
    params_to_update = []
    for name, param in torch_model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    return params_to_update


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class CNNTrain():
    def __init__(self, configpath="imgcls/imgcls.yaml") -> None:
        with open(configpath, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        print(config)

        # model output
        self.device = get_device()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(model.fc.in_features, config['model']['output']['num_classes'])

        self.probability = torch.nn.Softmax(dim=1)
        self.model = model

        params_to_update = get_params_requires_grad(model)

        if config['model']['cnn']['optimizer']['type'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                params_to_update,
                lr=config['model']['cnn']['optimizer']['lr'],
                momentum=config['model']['cnn']['optimizer']['momentum'],
            )
            print(self.optimizer)

        self.num_epochs = config['model']['cnn']['num_epochs']

        # model input
        #   todo: move transforms to config....
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.trainset = ImgClsDataset(
            dataset_file=config['model']['input']['train']['file'],
            shuffle=True,
            load_folds=config['model']['input']['train']['folds'],
            transform=train_transforms,
        )
        self.valset = ImgClsDataset(
            dataset_file=config['model']['input']['val']['file'],
            shuffle=True,
            load_folds=config['model']['input']['val']['folds'],
            transform=val_transforms,
        )

        self.trainiter = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=config['model']['input']['train']['batch_size'],
            shuffle=True,
        )
        
        self.valiter = torch.utils.data.DataLoader(
            self.valset,
            batch_size=config['model']['input']['val']['batch_size'],
            shuffle=False,
        )

        self.location = config['model']['location']
        
        with open(os.path.join(self.location, "config.yaml"), "w") as stream:
            try:
                yaml.safe_dump(config, stream)
            except yaml.YAMLError as exc:
                print(exc)

    def train_epoch(self):
        print("Started train one epoch.")
        num_batches = len(self.trainiter)

        self.model.train()  # set model to train mode
        size = len(self.trainset)

        train_loss = 0
        for batch_index, (X, y) in enumerate(self.trainiter):
            X, y = X.to(self.device), y.to(self.device)
            loss = self.backprop(X, y)
            train_loss += loss.item()

            if batch_index % 100 == 0:
                _loss, current = loss.item(), (batch_index + 1) * len(X)
                print(f"loss: {_loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        train_loss /= num_batches
        print(f"Done train one epoch; avg_loss: {train_loss}.")
        return train_loss
    
    def validation(self):
        print("Started validation.")
        num_batches = len(self.valiter)
        
        self.model.eval() # set model to evaluation mode

        loss = 0.0
        error_rate = 0.0
        with torch.no_grad():
            for batch_index, (X, y) in enumerate(self.valiter):
                X, y = X.to(self.device), y.to(self.device)
                logits = self.forward(X)
                loss += self.criterion(logits, y).item()
                error_rate += self.mean_error_rate(logits, y)

        loss /= num_batches
        error_rate /= num_batches

        print(f"Validation stats: \n ErrorRate: {(100 * error_rate):>0.1f}%, AvgLoss: {loss:>8f} \n")

        return loss
    
    def mean_error_rate(self, logits, y):
        err = y.argmax(dim=1) != self.probability(logits).argmax(dim=1)
        err = err.type(torch.float)
        err = torch.mean(err).item()
        return err
    
    def forward(self, X):
        logits = self.model(X)
        return logits
    
    def backprop(self, X, y):
        logits = self.forward(X)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self):
        print("Train start.")

        train_losses = []
        val_losses = []

        self.model.to(self.device)
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss = self.train_epoch()
            val_loss = self.validation()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        model_file = os.path.join(self.location, "imgcls.pt")
        torch.save(self.model, model_file)
        
        pd.DataFrame({
            'train_losses': train_losses,
            'val_losses': val_losses,
        }).to_csv("loss.csv")

        print("Train end.")


if __name__ == "__main__":
    train = CNNTrain()
    train.train()