import torch
from slicegpt.utils import cleanup_memory
from .kron_utils import kron_project_weighted, kron_merge
from .monarch_utils import monarch_project_weighted, monarch_merge


def list_dot(Q, W, right=False):
    if right:
        return [W[i] @ Q for i in range(len(W))]
    return [Q @ W[i] for i in range(len(W))]


def weighted_orth_procrustes(Wout, Wout_appr, Xout, Win, Win_appr, Xin, S=None, n_iters=200,
                             use_exp=False, cut_values=False, use_cg=True,
                             Q_skip=None, Xskip=None, Q_prev=None):
    # loss = |Xout Wout Q - Xout Wout_appr| + |Xin Q Win_appr - Xin Win| + |Xskip Q_prev Q - Xskip Q_skip|
    if Q_skip is not None:
        assert Xskip is not None and Q_prev is not None
    if Win is not None:
        N = Win[0].shape[0]
        device = Win[0].device
    else:
        N = Wout[0].shape[1]
        device = Wout[0].device
    eye = torch.eye(N, device=device)
    if Xout is not None:
        Y = [Xout[0] @ Wout_appr[0]]
        L = [Xout[0] @ Wout[0]]
    else:
        Y = Wout_appr
        L = Wout
    R = [eye]
    if Win is not None:
        Y += [Xin[i] @ Win[i] for i in range(len(Xin))]
        R += Win_appr
        L += Xin
        if cut_values:
            Y, L, R = Y[:-1], L[:-1], R[:-1]
        alpha = torch.mean(torch.tensor([torch.norm(y) for y in Y[1:]])) / torch.norm(L[0])
        Y[0] *= alpha
        L[0] *= alpha
    if Q_skip is not None:
        Y += [Xskip[0] @ Q_skip]
        L += [Xskip[0] @ Q_prev]
        R += [eye]
    del Wout, Wout_appr, Xout, Win, Win_appr, Xin, Q_skip, Xskip, Q_prev
    cleanup_memory()

    # gradient descent
    errs = []
    length = len(L)
    if S is None:
        S = eye.clone()
    lr = 1.0
    gamma = 2.0
    max_iters = 20
    d = 0
    prev_grad_norm = None

    def cnt_loss(S):
        if use_exp:
            Q = torch.linalg.matrix_exp(S - S.T)  # matrix exponent
        else:
            Q = 2 * torch.linalg.inv(S - S.T + eye) - eye  # cayley transform
        return sum([torch.norm(L[i] @ Q @ R[i] - Y[i]) for i in range(length)])

    old_loss, S_old = None, None
    flag = False
    for _ in range(n_iters):
        S.requires_grad = True
        S.retain_grad()
        loss = cnt_loss(S)
        if old_loss is not None and loss > old_loss:
            with torch.no_grad():
                c = 0
                while loss > old_loss and c < max_iters:
                    lr /= gamma
                    S = S_old + lr * d
                    c += 1
                    loss = cnt_loss(S)
                if c >= max_iters:
                    flag = True
                    break
            S.requires_grad = True
            S.retain_grad()
            loss = cnt_loss(S)
        loss.backward()
        old_loss = loss.item()
        errs.append(old_loss)
        with torch.no_grad():
            if use_cg:
                grad = S.grad
                grad_norm = torch.norm(S.grad) ** 2
                beta = grad_norm / prev_grad_norm if prev_grad_norm is not None else 0
                prev_grad_norm = grad_norm
                S_old = S
                d = -grad + beta * d
                S = S_old + lr * d
            else:
                S_old = S
                d = -S.grad / torch.norm(S.grad)
                S = S_old + lr * d
    if flag:
        S = S_old
    errs.append(cnt_loss(S).item())
    if len(errs) >= 2:
        print('weighted procrustes errs', errs[0], errs[-1], flush=True)
    S = S.detach()
    if use_exp:
        Q = torch.linalg.matrix_exp(S - S.T)  # matrix exponent
    else:
        Q = 2 * torch.linalg.inv(S - S.T + eye) - eye  # cayley transform
    del L, R, Y, S_old, errs, eye
    cleanup_memory()
    return Q, S



def weighted_procrustes_als(Wout, Xout, Win, Xin, params_out, params_in, cfg, layer_cfg, device, cut_values,
                            diag_emb=None, Xskip=None, Q_prev=None):
    # |Xout Wout Q - Xout Wout'| + |Xin Win - Xin Q Win'| -> min
    # |Xout Wout Q - Xout Wout'| + |(Xin Q) (Q.T Win) - (Xin Q) Win'| -> min
    Xout_inv = None
    if Xout is not None:
        Xout, Xout_inv = Xout
    Xin, Xin_inv = Xin
    if layer_cfg.cfg.decomposition == 'kronecker':
        project = kron_project_weighted
        merge = kron_merge
    elif layer_cfg.cfg.decomposition == 'monarch':
        project = monarch_project_weighted
        merge = monarch_merge
    else:
        raise ValueError('unknown decomposition')
    if Wout is not None:
        Wout = [w.to(device, torch.float32) for w in Wout]
    if Xout is not None:
        Xout = [w.to(device, torch.float32) for w in Xout]
    if Win is not None:
        Win, Xin = [w.to(device, torch.float32) for w in Win], [w.to(device, torch.float32) for w in Xin]
    if Win is not None:
        assert Win[0].shape[0] == Xin[0].shape[1]
        N = Win[0].shape[0]
    else:
        N = Wout[0].shape[1]
    if Wout is not None and Win is not None:
        assert Wout[0].shape[1] == Win[0].shape[0]
    Q = torch.eye(N, device=device)
    if cfg.test:
        Q = torch.randn((N, N), device=device)
        Q, _ = torch.linalg.qr(Q, mode='complete')
    Q_skip, S, Win_appr, Wout_appr = None, None, None, None
    S = None
    for _ in range(cfg.als_iters):
        # project
        if Wout is not None:
            Wout_appr = project(list_dot(Q, Wout, right=True), params_out, Xout, inv=Xout_inv, inp_x_out=True,
                                approx_iters=cfg.approx_iters, W_appr=None)
        if Win is not None:
            Win_appr = project(list_dot(Q.T, Win), params_in, list_dot(Q, Xin, right=True),
                               inv=list_dot(Q, list_dot(Q, Xin, right=True)),
                               inp_x_out=True, approx_iters=cfg.approx_iters, W_appr=None)
        # solve orthogonal procrustes problem
        Q, S = weighted_orth_procrustes(Wout if diag_emb is None else [Wout[0] * diag_emb[:, None]],
                                        merge(Wout_appr, params_out), Xout, Win,
                                        merge(Win_appr, params_in), Xin, S=S,
                                        n_iters=cfg.gd_iters, cut_values=cut_values, use_cg=cfg.use_cg,
                                        Q_skip=Q_skip[0] if Q_skip is not None else None, Xskip=Xskip, Q_prev=Q_prev)
    # project
    if Wout is not None:
        Wout_appr = project(list_dot(Q, Wout, right=True), params_out, Xout, inv=Xout_inv, inp_x_out=True,
                            approx_iters=cfg.approx_iters, W_appr=None)
    if Win is not None:
        Win_appr = project(list_dot(Q.T, Win), params_in, list_dot(Q, Xin, right=True),
                           inv=list_dot(Q, list_dot(Q, Xin, right=True)),
                           inp_x_out=True, approx_iters=cfg.approx_iters, W_appr=None)
    del Xin, Xout, Xout_inv, Xin_inv, Win, Wout, params_in, params_out, S, Xskip, Q_prev
    cleanup_memory()
    return Q.T, Wout_appr, Win_appr, Q_skip
