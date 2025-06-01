import argparse
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler, ModelEma
from timm.scheduler import create_scheduler
import matplotlib.pyplot as plt
from functools import partial
import sys
import os
sys.path.append(os.path.abspath(".."))
from torch.utils.data import DataLoader
from Engine import train_one_epoch, evaluate
from util import load_checkpoint, SmoothedValue
import torch
from timm.models import create_model
from timm.models.vision_transformer import _cfg
from functools import partial
from Model import VisionTransformer
from timm.models.registry import register_model
from timm.models import load_checkpoint
from torch import nn
from dataset import LoadDatasets, transform_video_val
from util import MetricLogger, optimizer_to



@register_model
def vit(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay = None,  **kwargs):
    model = VisionTransformer(
        img_size=[224, 224],
        patch_size=[1, 2],
        tubelet =[1, 2],
        in_chans=3,
        num_classes=97,
        embed_dim=[224, 448],
        depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
        num_heads=[7, 7],
        mlp_ratio=[3, 3, 1],
        qkv_bias=True,
        weight_pretrained=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    model.default_cfg = _cfg
    return model

vit = create_model(
        model_name= 'vit',
        pretrained = False)
def get_args_parser():
    parser = argparse.ArgumentParser('CrossViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--data-path', default='/home/tludemo/sign_language/dataset', type=str,
                        help='dataset path (default: /home/tludemo/sign_language/dataset)')
        # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
     # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--smoothing', type=float, default=0.1, 
                         help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('-f')
    return parser



def main(args, tuning = True):
    device = 'cuda'
    device = torch.device(device)
    criterion = LabelSmoothingCrossEntropy(0.1)
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at {train_path}. Please check the path.")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation dataset not found at {val_path}. Please check the path.")
    if not os.path.exists('/home/tludemo/sign_language/weight'):
        os.makedirs('/home/tludemo/sign_language/weight')
    dataset_train = LoadDatasets(path = train_path, transform_video=transform_video_val)
    dataset_val = LoadDatasets(path = val_path, transform_video=transform_video_val)
    train_batch_datasets = DataLoader(dataset_train, batch_size=6, shuffle=True)
    val_batch_datasets = DataLoader(dataset_val, batch_size=1, shuffle=False)
    optimizer = create_optimizer_v2(
        model = vit,
        optimizer_name='adamw',
        filter_bias_and_bn=True,
        learning_rate = 5e-6,
        weight_decay = 0.05,
        momentum= 0.99,
        eps = 1e-8,
    )
    model_ema = ModelEma(
            vit,
            decay= 0.9999,
            device='cuda:0',
            resume='')
    lr_scheduler, _ = create_scheduler(args, optimizer= optimizer)
    max_accuracy = 0
    start_epoch = 0
    vit.to('cuda')
    loss_scaler = NativeScaler()
    if tuning == True:
        checkpoint = torch.load(f = '/home/tludemo/sign_language/weight/best_model_demo.pt', map_location='cpu', weights_only=False )
        # load_checkpoint(vit, checkpoint['model'])
        vit.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        max_accuracy = checkpoint['max_accuracy']
        model_ema = model_ema.update(vit)
        optimizer_to(optimizer, 'cuda:0')
        tuning = False
    vit.eval()
    metric_logger = MetricLogger(delimiter="  ")

   
    for epoch in range(start_epoch+1 , 100):
            _, precision_matrices, inputs = train_one_epoch(model  = vit, criterion=criterion, data_loader=val_batch_datasets, optimizer=optimizer, device=device, epoch=epoch, model_ema=model_ema, mixup_fn =  None, max_norm=args.clip_grad,  amp=False)
            lr_scheduler.step(epoch)
            val_accuracy = evaluate(test_data_loader=val_batch_datasets, model=vit, device='cuda', amp=False)
            max_accuracy = max(val_accuracy['acc1'], max_accuracy)
            
            if max_accuracy == val_accuracy['acc1']:
                state_dict ={
                    'model': vit.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_accuracy': max_accuracy,          
                    }
                torch.save(state_dict, 
                    '/home/tludemo/sign_language/weight/test.pt')
            else:
                state_dict ={
                    'model': vit.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'va_accuracy': val_accuracy['acc1'],          
                    }
                torch.save(state_dict, 
                    '/home/tludemo/sign_language/weight/a.pt')


parser = argparse.ArgumentParser('CrossViT training and evaluation script', parents=[get_args_parser()])
args, unknown = parser.parse_known_args() 
main(args)
   # /home/tludemo/trunghm/SL-PTIT-50/yeu/N5_yeu_15.mp4
    