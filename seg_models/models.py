import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from seg_utils.utils import ToTensor
from seg_models.esp_model import ESPNet, ESPNet_Encoder
from seg_models.stdc_model import BiSeNet

from seg_utils.dataset_utils import dataset, split_dataset

def train_encoder(args):
    model = ESPNet_Encoder(1, p=2, q=8)
    model.to(args.device)
    save_path = 'espnet_p_2_q_8.pth'

    criterion = smp.losses.DiceLoss('binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Dataset Loader
    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(), #scaleIn = 8
    ])

    datasets = dataset(args, args.root, args.mode, transform=transform)
    train_dataset, test_dataset = split_dataset(datasets, args.split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Train
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        target = target.unsqueeze(1)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_path)


def load_model(args, model_name):
    if model_name == 'DeepLabv3':
        ENCODER = 'resnet101'
        ENCODER_WEIGHTS = 'imagenet'
        ACTIVATION = 'sigmoid'

        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            activation=ACTIVATION,
        )
    elif model_name == 'ESPNet':
        args.model = 'ESPNet_Encoder'
        train_encoder(args)
        
        encoder_path = 'espnet_p_2_q_8.pth'
        args.model = 'ESPNet'
        model = ESPNet(1, p=2, q=8)
    elif model_name == 'STDC':
        model = BiSeNet(backbone='STDCNet813', n_classes=1, pretrain_model='checkpoints/STDCNet813M_73.91.tar', use_boundary_2=False, use_boundary_4=False, use_boundary_8=True,  use_boundary_16=False, use_conv_last=False)

    return model