import math
from typing import Iterable, Optional
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.utils import accuracy
from timm.utils import ModelEma
# from einops import rearrange
from util import MetricLogger, SmoothedValue


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,  max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=False,
                    finetune=False, model_ema = None
                    ):
    if finetune:
        model.train(not finetune)
    else:
        model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    for data_iter_step, (samples, targets, _, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)):
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        batch_size = targets.size(0)
        print(samples.getType())
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        samples = samples.float()
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast(enabled = amp):
            outputs = model.forward(samples)
            loss = criterion(outputs, targets)
            # loss_auxilary = criterion(auxilary_outputs, targets)
            # loss = 0.3 * loss_auxilary + 2 * loss_main
        loss_value = loss.detach().item()
        acc1, acc2 = accuracy(outputs, targets, topk=(1, 2))
        metric_logger.meters['acc1'].update(acc1.item(), n=1)
        metric_logger.meters['acc2'].update(acc2.item(), n=1)
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        amp = False
        if amp:
            pass
        else:
            loss.backward(create_graph=is_second_order)
            max_norm =0
            if max_norm is not None and max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    with open('out.txt', 'a') as f:
        f.write( '* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top2=metric_logger.acc2,
            losses=metric_logger.loss))
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top2=metric_logger.acc2,
            losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(test_data_loader, model, device, amp=False):
    criterion = nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # model.train()
    outputs = []
    targets = []
    pred = []
    gt = []
    # print_freq = 100
    for batch in metric_logger.log_every(test_data_loader, 100, header):
        images = batch[0]
        target = batch[1]
        samples = images.to(device, non_blocking=True)
        targets = target.to(device, non_blocking=True)
        samples = samples.float()
        # compute output
        with torch.cuda.amp.autocast():
            outputs = model.forward(samples)
            loss = criterion(outputs, targets)
        # roc_auc.update(outputs, target)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        acc1, acc2 = accuracy(outputs, targets, topk=(1, 2))
        metric_logger.meters['acc1'].update(acc1.item(), n=1)
        metric_logger.meters['acc2'].update(acc2.item(), n=1)
       
    # result = roc_auc.compute()
    # print(result)
    # with open('roc_i3d.txt', 'a') as f:
    #     f.write(str(result))

    metric_logger.synchronize_between_processes()
    with open('out.txt', 'a') as f:
        f.write( '* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top2=metric_logger.acc2,
            losses=metric_logger.loss))
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top2=metric_logger.acc2,
            losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}