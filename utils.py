import torch
import torch.optim as optim
import torch.nn as nn
import models
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(args):
    input_train = torch.load('./data/'+args.dataset+'/train/input_train.pt')
    target_train = torch.load('./data/'+args.dataset+'/train/target_train.pt')
    if args.test_val_train == 'val':
        input_val = torch.load('./data/'+args.dataset+'/val/input_val.pt')
        target_val = torch.load('./data/'+args.dataset+'/val/target_val.pt')
    elif args.test_val_train == 'train':
        input_val = input_train
        target_val = target_train

    global train_shape_in , train_shape_out, val_shape_in, val_shape_in
    train_shape_in = input_train.shape
    train_shape_out = target_train.shape
    val_shape_in = input_val.shape
    val_shape_out = target_val.shape
    input_train = torch.log1p(input_train)
    target_train = torch.log1p(target_train)
    input_val = torch.log1p(input_val)
    target_val = torch.log1p(target_val)
   
    train_data = TensorDataset(input_train,  target_train)
    val_data = TensorDataset(input_val, target_val)
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) 
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    return [train, val, train_shape_in, train_shape_out, val_shape_in, val_shape_out]


def load_model(args):
    if args.model == 'resnet':
        model = models.ResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks,
                              upsampling_factor=args.upsampling_factor, dim=args.dim_channels)
    elif args.model.lower() == 'rmresnet':
        rcan   = models.RCAN(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks,
                             upsampling_factor=args.upsampling_factor, dim=args.dim_channels,n_groups=5, reduction=16, res_scale=0.1).to(device)
        resnet = models.ResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks,
                               upsampling_factor=args.upsampling_factor, dim=args.dim_channels).to(device)
        model  = models.RMResNet(rcan=rcan, resnet=resnet,tau=args.tau, k=float(args.gate_k), feather_sigma=float(args.feather))
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#trainable params =', n_train)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = model.to(device)
    return model

def get_optimizer(args, model):
    lr = args.lr
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

def get_loss(output, true_value, in_val, args, model=None):
    loss_name = str(args.loss).lower()
    if loss_name == 'mse':
        return torch.nn.functional.mse_loss(output, true_value)
    elif loss_name == 'mae':
        return torch.nn.functional.l1_loss(output, true_value)
    elif loss_name == 'huber':
        huber = torch.nn.HuberLoss(delta=getattr(args, 'delta', 1.0), reduction='mean')
        return huber(output, true_value)
    elif loss_name == 'rmloss':
        crit = RMloss(tau=getattr(args, 'tau', 0.03), w_val=1.0, w_grad=0.15, w_mask=0.05)
        return crit(output, true_value, mask_prob=getattr(model, 'last_mask', None))
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

def process_for_training(inputs, targets): 
    inputs = inputs.to(device)            
    targets = targets.to(device)
    return inputs, targets

def process_for_eval(outputs, targets):
    outputs = torch.expm1(outputs)
    targets = torch.expm1(targets)
    return outputs, targets

def sobel_kernels():
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)/4.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)/4.0
    return kx.view(1,1,3,3), ky.view(1,1,3,3)


def gradient_loss(y_pred, y_true, edge_weight=None):
    kx, ky = sobel_kernels()
    device = y_pred.device
    kx = kx.to(device); ky = ky.to(device)

    def grad_map(y):
        gx = F.conv2d(y, kx, padding=1)
        gy = F.conv2d(y, ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-6)

    gp = grad_map(y_pred)
    gt = grad_map(y_true)
    diff = torch.abs(gp - gt)
    if edge_weight is not None:
        diff = diff * edge_weight
    return diff.mean()

class RMloss(nn.Module):
    def __init__(self, tau=0.03, w_val=1.0, w_grad=0.15, w_mask=0.05):
        super().__init__()
        self.tau    = float(tau)
        self.w_val  = w_val
        self.w_grad = w_grad
        self.w_mask = w_mask
    def forward(self, pred, gt, mask_prob=None):
        l_val  = F.l1_loss(pred, gt)
        l_grad = gradient_loss(pred, gt, edge_weight=None)

        if mask_prob is not None:
            gt_mask = (torch.expm1(gt) > self.tau).float()
            bce = F.binary_cross_entropy(mask_prob.clamp(1e-6, 1-1e-6), gt_mask)
            inter = (mask_prob * gt_mask).sum(dim=(1,2,3))
            union = (mask_prob + gt_mask - mask_prob * gt_mask).sum(dim=(1,2,3)) + 1e-6
            iou = (inter / union).mean()
            l_mask = 0.5 * bce + 0.5 * (1.0 - iou)
        else:
            l_mask = torch.zeros((), device=pred.device)

        return self.w_val * l_val + self.w_grad * l_grad + self.w_mask * l_mask


