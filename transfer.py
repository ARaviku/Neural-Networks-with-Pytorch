
import torch
import torchvision.models as models
from dataset import DogDataset
from train import train


def load_pretrained(num_classes=5):
    """
    Load a ResNet-18 model from `torchvision.models` with pre-trained weights. Freeze all the parameters besides the
    final layer by setting the flag `requires_grad` for each parameter to False. Replace the final fully connected layer
    with another fully connected layer with `num_classes` many output units.
    Inputs:
        - num_classes: int
    Returns:
        - model: PyTorch model
    """
    resnet18 = models.resnet18(pretrained=True)
    for parameter in resnet18.parameters():
        parameter.requires_grad = False
    resnet18.fc.weight.requires_grad = True
    resnet18.fc = nn.Linear(512, num_classes)
    return resnet18


if __name__ == '__main__':
    config = {
            'dataset_path': 'data/images/dogs',
            'batch_size': 4,
            'if_resize': False,
            'ckpt_path': 'checkpoints/transfer',
            'plot_name': 'Transfer',
            'num_epoch': 5,
            'learning_rate': 1e-3,
            'momentum': 0.9,
        }
    dataset = DogDataset(config['batch_size'], config['dataset_path'],config['if_resize'])
    model = load_pretrained()
    train(config, dataset, model)
