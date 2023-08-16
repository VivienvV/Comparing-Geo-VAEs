" Code taken from https://github.com/nicola-decao/s-vae-pytorch "
import torch


def tangent_proj(p, w):
    # p.shape [B, d, ...]
    # w.shape [B, d, ...]
    return w - torch.einsum('bi...,bj...,bi...->bj...',p,p,w)

def log(p, q):
    arccos_pq = torch.arccos(torch.einsum('bi...,bi...->b...', p, q)[:, None])  # [B, 1, ...]
    proj_pq = tangent_proj(p, q - p)
    proj_pq_norm = torch.linalg.norm(proj_pq, dim=1, keepdim=True)
    return arccos_pq * proj_pq / proj_pq_norm

def exp(p, v):
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
    return torch.cos(v_norm) * p + torch.sin(v_norm) * v / (0.00000001 + v_norm)

p = torch.tensor([[1.,0.,0.]])
w = torch.tensor([[1.,1.,0.]])
q = torch.tensor([[0.,1.,0.]])