import torch
from slicegpt.utils import cleanup_memory
from .kron_utils import kron_project, kron_merge
from .monarch_utils import monarch_merge, monarch_project


def list_dot(Q, W, right=False):
    if right:
        return [W[i] @ Q for i in range(len(W))]
    return [Q @ W[i] for i in range(len(W))]


@torch.no_grad()
def orth_procrustes(A, B):
    # |QA - B|_F -> min s.t. Q^TQ = I
    # coef = 1 / (torch.norm(A) * torch.norm(B)) ** 0.5
    # A, B = A * coef, B * coef
    U, s, VT = torch.linalg.svd(B @ A.T, full_matrices=False)
    del A, B
    Q = U @ VT
    del U, s, VT
    cleanup_memory()
    return Q


def procrustes_als(W, params, cfg, name='als', layer_cfg=None,
                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if layer_cfg.cfg.decomposition == 'kronecker':
        project = kron_project
        merge = kron_merge
    elif layer_cfg.cfg.decomposition == 'monarch':
        project = monarch_project
        merge = monarch_merge
    else:
        raise ValueError('unknown decomposition')
    W = [w.to(device, dtype=torch.float32) for w in W]
    W_d = []
    for i in range(len(W)):
        if 'diag' in params[i].keys():
            if W[i].shape[0] == params[i]['diag'].shape[0]:
                W_d.append(W[i] * params[i]['diag'][:, None])
            else:
                W_d.append(W[i] * params[i]['diag'][None, :])
        else:
            W_d.append(W[i])
    W_d = torch.hstack(W_d)
    Q = torch.eye(W[0].shape[0], device=device)
    if cfg.test:
        Q, _ = torch.linalg.qr(torch.randn((W[0].shape[0], W[0].shape[0]), device=device), mode='complete')
    norm = torch.norm(torch.hstack(W)).item()
    errs = []
    cnt = 0
    W_appr = None
    while cnt < cfg.als_iters and (cnt == 0 or abs(errs[-1] - errs[-2]) > cfg.eps):
        # decompose
        W_appr = merge(project(list_dot(Q, W), params), params)
        errs.append(torch.norm(Q @ W_d - torch.hstack(W_appr)).item() / norm)
        # solve orthogonal procrustes
        Q = orth_procrustes(W_d, torch.hstack(W_appr))
        errs.append(torch.norm(Q @ W_d - torch.hstack(W_appr)).item() / norm)
        cnt += 1
    print(name, errs, flush=True)
    del errs, params, W, W_d, cfg, W_appr
    cleanup_memory()
    return Q
