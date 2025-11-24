from utils import process_for_training, load_model, get_optimizer, process_for_eval, get_loss
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import csv
from torch.utils.data import TensorDataset
from torchmetrics.functional import multiscale_structural_similarity_index_measure, structural_similarity_index_measure
from skimage import transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(args, data):
    model = load_model(args)
    print('#params:', sum(p.numel() for p in model.parameters()))
    optimizer = get_optimizer(args, model)
    best = np.inf
    for epoch in range(args.epochs):
        running_loss = 0    
        
        for (inputs,  targets) in data[0]:          
            inputs, targets = process_for_training(inputs, targets)
            loss = optimizer_step(model, optimizer, inputs, targets, args)
            running_loss += loss
        loss = running_loss/len(data[0])
        print('Epoch {}, Train Loss: {:.5f}'.format(epoch+1, loss))
        val_loss = validate_model(model, data[1], args)
        print('Val loss: {:.5f}'.format(val_loss))
        checkpoint(model, val_loss, best, args, epoch)
        best = np.minimum(best, val_loss)      
    scores = evaluate_model( data, args)
    
def optimizer_step(model, optimizer, inputs, targets, args):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = get_loss(outputs, targets, inputs, args, model=model)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate_model(model, data, args):
    model.eval()
    running_loss = 0      
    for i, (inputs, targets) in enumerate(data):     
        inputs, targets = process_for_training(inputs, targets)
        outputs = model(inputs)
        loss = get_loss(outputs, targets, inputs, args, model=model)
        running_loss += loss.item()
    loss = running_loss/len(data)
    model.train()
    return loss


def checkpoint(model, val_loss, best, args, epoch):
    if val_loss < best:
        checkpoint = {'model': model,'state_dict': model.state_dict()}
        torch.save(checkpoint, './models/'+args.model_id+'.pth')
        
def evaluate_model(data, args):
    model = load_model(args)
    load_weights(model, args.model_id)
    model.eval()
    val_loader = data[1]
    N = len(val_loader.dataset)
    full_pred = None 
    k = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, unit="batch"):
            inputs, targets = process_for_training(inputs, targets)
            outputs = model(inputs)
            outputs, targets = process_for_eval(outputs, targets)
            bs = outputs.shape[0]
            if full_pred is None:
                full_pred = torch.zeros((N, *outputs.shape[1:]), dtype=outputs.dtype)
            full_pred[k:k+bs, ...] = outputs.detach().cpu()
            k += bs
    torch.save(
        full_pred,
        f'./data/prediction/{args.dataset}_{args.model_id}_{args.test_val_train}.pt'
    )
    calculate_scores(args)

    
def calculate_scores(args):
    input_val = torch.load('./data/'+args.dataset+'/'+ args.test_val_train+'/input_'+ args.test_val_train+'.pt')
    target_val = torch.load('./data/'+args.dataset+'/'+ args.test_val_train+'/target_'+ args.test_val_train+'.pt')
    val_data = TensorDataset(input_val, target_val)
    max_val = target_val.max()
    min_val = target_val.min()
    mse = 0
    mae = 0
    ssim = 0
    mean_bias = 0
    mean_abs_bias = 0
    mass_violation = 0
    ms_ssim = 0
    corr = 0
    neg_mean = 0
    neg_num = 0
    
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    
    pred = torch.load('./data/prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'.pt')
    pred = pred.detach().cpu()
    j = 0    
    for i, (lr, hr) in enumerate(val_data):
        pr = pred[i, 0, ...]   
        gt = hr[0, ...]       
        mse += l2_crit(pr, gt).item()
        mae += l1_crit(pr, gt).item()
        mean_bias += torch.mean(gt - pr)
        mean_abs_bias += torch.abs(torch.mean(gt - pr))
        corr += pearsonr(pr.flatten(), gt.flatten())
        pr_4d = pr.unsqueeze(0).unsqueeze(0)           
        gt_4d = gt.unsqueeze(0).unsqueeze(0)           
        ms_ssim += multiscale_structural_similarity_index_measure(
            pr_4d, gt_4d,
            data_range=max_val - min_val,
            kernel_size=11, betas=(0.2856, 0.3001, 0.2363)
        )
        ssim += structural_similarity_index_measure(
            pr_4d, gt_4d,
            data_range=max_val - min_val, kernel_size=11
        )
        neg_num += int((pr < 0).sum().item())
        neg_mean += float((pr[pr < 0].sum() / (pr.numel() if pr.numel() else 1)).item())
        im = lr[0, ...]
        pr_down = transform.downscale_local_mean(pr.numpy(), (args.upsampling_factor, args.upsampling_factor))
        im_np   = im.numpy()
        mass_violation += np.mean(np.abs(pr_down - im_np))

    mse *= 1/input_val.shape[0]
    mae *= 1/input_val.shape[0]
    ssim *= 1/input_val.shape[0]
    mean_bias *= 1/input_val.shape[0]
    mean_abs_bias *= 1/input_val.shape[0]
    corr *= 1/input_val.shape[0]
    ms_ssim *= 1/input_val.shape[0]
    neg_mean *= 1/input_val.shape[0]
    mass_violation *= 1/input_val.shape[0]
    psnr = calculate_pnsr(mse, target_val.max() )   
    rmse = torch.sqrt(torch.Tensor([mse])).numpy()[0]
    ssim = float(ssim.numpy())
    ms_ssim =float( ms_ssim.numpy())
    psnr = psnr.numpy()
    corr = float(corr.numpy())
    mean_bias = float(mean_bias.numpy())
    mean_abs_bias = float(mean_abs_bias.numpy())
    scores = {'MSE':mse, 'RMSE':rmse, 'PSNR': psnr[0], 'MAE':mae, 'SSIM':ssim,  'MS SSIM': ms_ssim, 'Pearson corr': corr, 'Mean bias': mean_bias, 'Mean abs bias': mean_abs_bias, 'Mass_violation': mass_violation, 'neg mean': neg_mean}
    print(scores)
    create_report(scores, args)


def calculate_pnsr(mse, max_val):
    return 20 * torch.log10(max_val / torch.sqrt(torch.Tensor([mse])))
                                            
def create_report(scores, args):
    args_dict = args_to_dict(args)
    args_scores_dict = args_dict | scores
    save_dict(args_scores_dict, args)
    
def args_to_dict(args):
    return vars(args)
    
                                            
def save_dict(dictionary, args):
    w = csv.writer(open('./data/'+args.model_id+'.csv', 'w'))      
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])

def load_weights(model, model_id):
    PATH = './models/'+model_id+'.pth'
    checkpoint = torch.load(PATH) # ie, model_best.pth.tar
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda')
    return model

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


