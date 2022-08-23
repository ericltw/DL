import torch
import torch.nn as nn


# Custom weights initialization called on Generator and Discriminator.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# Compute the current classification accuracy.
def compute_acc(out, onehot_labels):
    batch_size = out.size(0)
    acc = 0
    total = 0
    for i in range(batch_size):
        k = int(onehot_labels[i].sum().item())
        total += k
        outv, outi = out[i].topk(k)
        lv, li = onehot_labels[i].topk(k)
        for j in outi:
            if j in li:
                acc += 1
    return acc / total


def MBCE(input, target, esp=1e-19):
    loss = - torch.mean(target * torch.log(input.clamp_min(esp))) - torch.mean(
        (1 - target) * torch.log((1 - input).clamp_min(esp)))
    return loss
