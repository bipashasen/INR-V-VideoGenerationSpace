import torch
import torch.nn.functional as F

import diff_operators
import modules
import torch.nn as nn

from lpips import LPIPS

class LPIPSLoss(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.perceptual_weight = 0.7 #1.0

        self.perceptual_loss = LPIPS().eval()

        self.shape = (-1, img_dim[0], img_dim[1], 3)

    def forward(self, rec_loss, targets, reconstructions):
        def transform_dims(x):
            return x.view(self.shape).permute(0, 3, 1, 2)

        targets = transform_dims(targets).float()
        reconstructions = transform_dims(reconstructions).float()
        
        rec_loss = transform_dims(rec_loss)

        p_loss = self.perceptual_loss(targets.contiguous(), reconstructions.contiguous())

        assert len(p_loss) == len(rec_loss)

        rec_loss = rec_loss.contiguous().view(len(p_loss), -1).mean(1).view(p_loss.shape)
        
        rec_loss = rec_loss + self.perceptual_weight * p_loss

        return torch.mean(rec_loss)

def kl_loss2(mu, log_var):

    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

def kl_loss(mu, log_var, std=0.01):
    std = log_var.mul(0.5).exp_()

    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std) * std)
    q = torch.distributions.Normal(mu, std)

    return 0.05 * torch.distributions.kl_divergence(p, q).mean()

def cross_entropy(pred, tgt):
    ce = nn.CrossEntropyLoss()

    return ce(pred, tgt)

def normalized_euclidean(input1, input2):
    normalized_input2 = ((0.5 * (input2 + 1.)) + 1.) * 2.

    return ((normalized_input2 * ((input1 - input2) ** 2)) + 1e-5)

def euclidean(input1, input2): # model-out, gt.

    return ((input1 - input2) ** 2)

def manhattan(input1, input2):
    return torch.abs(input1 - input2)

def image_mse(model_output, gt, mask=None, loss='euclidean', lpips=None, std=0.01):
    loss_fn = manhattan if loss == 'manhattan'\
        else (euclidean if loss == 'euclidean' else normalized_euclidean)

    targets, reconstructions = gt['img'], model_output['model_out']

    img_loss = loss_fn(reconstructions, targets)

    if lpips is not None:
        img_loss = lpips(img_loss, targets, reconstructions)
    else:
        img_loss = img_loss.mean()

    if mask is not None:
        img_loss = mask * img_loss

    return { 'img_loss': img_loss, }

def image_l1(mask, model_output, gt):
    if mask is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['img']).mean()}
    else:
        return {'img_loss': (mask * torch.abs(model_output['model_out'] - gt['img'])).mean()}


def image_mse_TV_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}


def image_mse_FH_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    img_hessian, status = diff_operators.hessian(rand_output['model_out'],
                                                 rand_output['model_in'])
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}


def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def image_hypernetwork_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_mse(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}


def function_mse(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt['gradients']).pow(2).sum(-1))
    return {'gradients_loss': gradients_loss}


def gradients_color_mse(model_output, gt):
    # compute gradients on the model
    gradients_r = diff_operators.gradient(model_output['model_out'][..., 0], model_output['model_in'])
    gradients_g = diff_operators.gradient(model_output['model_out'][..., 1], model_output['model_in'])
    gradients_b = diff_operators.gradient(model_output['model_out'][..., 2], model_output['model_in'])
    gradients = torch.cat((gradients_r, gradients_g, gradients_b), dim=-1)
    # compare them with the ground-truth
    weights = torch.tensor([1e1, 1e1, 1., 1., 1e1, 1e1]).cuda()
    gradients_loss = torch.mean((weights * (gradients[0:2] - gt['gradients']).pow(2)).sum(-1))
    return {'gradients_loss': gradients_loss}


def laplace_mse(model_output, gt):
    # compute laplacian on the model
    laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    # compare them with the ground truth
    laplace_loss = torch.mean((laplace - gt['laplace']) ** 2)
    return {'laplace_loss': laplace_loss}
