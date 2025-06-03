import torch
import torchmin
from einops import rearrange
from slicegpt.utils import cleanup_memory


def generate_perfect_shuffle(k, n):
    p = []
    for i in range(n):
        p.append((i % k) * (n // k) + i // k)
    return p


def generate_masks(p, kl, kr):
    rank_mask = torch.zeros(kl, kr, dtype=torch.long)

    br = len(p) // kr
    bl = len(p) // kl

    l_mask = [[0, 0, 0]] * len(p)
    r_mask = [[0, 0, 0]] * len(p)

    for i, pi in enumerate(p):
        r_ind = i // br
        l_ind = pi // bl
        r_mask[i] = [l_ind, r_ind, rank_mask[l_ind, r_ind].item()]
        l_mask[pi] = [l_ind, r_ind, rank_mask[l_ind, r_ind].item()]
        rank_mask[l_ind, r_ind] += 1

    return torch.tensor(l_mask, dtype=torch.long), torch.tensor(r_mask, dtype=torch.long), rank_mask


def inverse_permutation(p):
    if p is None:
        return None
    inv = [0] * len(p)
    for i, val in enumerate(p):
        inv[val] = i
    return inv


def gs_project_matrix(M, params, dtype, inp_x_out=True):
    M = M.to(torch.float32)
    kl, bl1, bl2 = params['kl'], params['bl1'], params['bl2']
    kr, br1, br2 = params['kr'], params['br1'], params['br2']
    use_pl, use_pr = params['use_pl'], params['use_pr']
    if inp_x_out:
        p = generate_perfect_shuffle(kl, kl * bl2)
    else:
        p = inverse_permutation(generate_perfect_shuffle(kr, kr * br1))
    pl, pr, pl_inv, pr_inv = None, None, None, None
    if use_pl:
        pl_inv = generate_perfect_shuffle(kl, M.shape[0])
        pl = inverse_permutation(pl_inv)
    if use_pr:
        pr_inv = generate_perfect_shuffle(kr, M.shape[1])
        pr = inverse_permutation(pr_inv)
    if not inp_x_out:
        pl, pl_inv, pr, pr_inv = pr_inv, pr, pl_inv, pl

    assert kl * bl2 == kr * br1
    assert M.shape[0] == kl * bl1 and M.shape[1] == kr * br2

    left = True
    if 'diag' in params.keys():
        if M.shape[0] == params['diag'].shape[0]:
            assert pl is None
            M = M * params['diag'][:, None]
        else:
            left = False
            assert pr is None
            M = M * params['diag'][None, :]

    if pr is not None:
        M = M[:, pr_inv]
    if pl is not None:
        M = M[pl_inv, :]

    M_tensor = rearrange(M, '(l i) (r j) -> l r i j', l=kl, r=kr)
    U, S, VT = torch.linalg.svd(M_tensor, full_matrices=False)

    L_svd = U * S[:, :, None, :].sqrt()  # l r i s
    R_svd = S[:, :, :, None].sqrt() * VT  # l r s j

    l_mask, r_mask, _ = generate_masks(p, kl, kr)

    L_cols = L_svd[l_mask[:, 0], l_mask[:, 1], :, l_mask[:, 2]]
    R_rows = R_svd[r_mask[:, 0], r_mask[:, 1], r_mask[:, 2], :]

    L = rearrange(L_cols, '(l s) i -> l i s', l=kl).to(dtype)
    R = rearrange(R_rows, '(r s) j -> r s j', r=kr).to(dtype)

    if 'diag' in params.keys():
        if left:
            L = L / params['diag'].reshape(kl, bl1)[:, :, None]
        else:
            R = R / params['diag'].reshape(kr, 1, br2)

    del M, M_tensor, U, S, VT, L_svd, R_svd, L_cols, R_rows, pl_inv, pr_inv, params
    cleanup_memory()

    return L, R, pl, p, pr


def multiply(X, L, R, pl, p, pr):
    """
    Returns batched product (pl L p R pr) @ X
    Input:
    X (..., kr*br1)
    L (kl, bl1, bl2)
    R (kr, br1, br2)
    Output:
    Y (...,kl * bl1)
    """
    batch_shape = X.shape[:-1]
    m = X.shape[-1]
    X = X.view(-1, m)
    kl, bl1, bl2 = L.shape
    kr, br1, br2 = R.shape
    assert kl * bl2 == kr * br1
    assert m == kr * br2

    if pr is not None:
        X = X[..., inverse_permutation(pr)]

    X = rearrange(X, 'b (k j) -> k j b', k=kr)  # (kr, br2, batch)
    X = torch.bmm(R, X)  # (kr, br1, batch)

    X = rearrange(X, 'k j b -> b (k j)')[..., inverse_permutation(p)]
    X = rearrange(X, 'b (k i) -> k i b', k=kl)  # (kl, bl2, batch)

    X = torch.bmm(L, X)  # (kl, bl1, batch)
    X = rearrange(X, 'k i b -> b (k i)')
    if pl is not None:
        X = X[..., pl]

    return X.reshape(*batch_shape, X.shape[-1])


def multiply_left(X, L, R, pl, p, pr):
    """
    Returns batched product X @ (pl L p R pr) 
    Input:
    X (..., kr*br1)
    L (kl, bl1, bl2)
    R (kr, br1, br2)
    Output:
    Y (...,kl * bl1)
    """
    batch_shape = X.shape[:-1]
    m = X.shape[-1]
    X = X.view(-1, m)
    kl, bl1, bl2 = L.shape
    kr, br1, br2 = R.shape
    assert kl * bl2 == kr * br1
    assert m == kl * bl1

    if pl is not None:
        X = X[..., inverse_permutation(pl)]

    X = rearrange(X, 'b (k j) -> k b j', k=kl)  # (kl, bl1, batch)
    X = torch.bmm(X, L)  # (kl, bl2, batch)

    X = rearrange(X, 'k b j -> b (k j)')[..., p]
    X = rearrange(X, 'b (k i) -> k b i', k=kr)  # (kr, br1, batch)

    X = torch.bmm(X, R)  # (kr, br2, batch)
    X = rearrange(X, 'k b i -> b (k i)')
    if pr is not None:
        X = X[..., pr]

    return X.reshape(*batch_shape, X.shape[-1])


def form_full(L, R, pl, p, pr):
    """
    Constracts dense matrix M from factors L, R and permutations
    """
    if R.shape[0] * R.shape[-1] < L.shape[0] * L.shape[1]:
        X = torch.eye(R.shape[0] * R.shape[-1], device=L.device, dtype=L.dtype)
        res = multiply(X, L, R, pl, p, pr).T
    else:
        X = torch.eye(L.shape[0] * L.shape[1], device=L.device, dtype=L.dtype)
        res = multiply_left(X, L, R, pl, p, pr)

    return res


@torch.no_grad()
def project_matrix(M, params, X=None, inp_x_out=True, approx_iters=0):
    assert inp_x_out
    if X is None:
        # project to M monarch matrices |M - pl L p R pr| -> min
        res = gs_project_matrix(M, params, dtype=torch.float32, inp_x_out=params['inp_x_out'])
        del M, params
        cleanup_memory()
        return res
    else:
        # project to M monarch matrices in weighted norm: |XM - X (pl L p R pr)| -> min
        errs = []
        assert inp_x_out
        L, R, pl, p, pr = project_matrix(M, params, X=None)
        kl, bl1, bl2 = params['kl'], params['bl1'], params['bl2']
        kr, br1, br2 = params['kr'], params['br1'], params['br2']
        if kr * bl1 * bl2 == 1 or kl * br1 * br2 == 1:
            return L, R, pl, p, pr
        Y = (X @ M)
        if pl is not None:
            X = X[:, inverse_permutation(pl)]
        if pr is not None:
            Y = Y[:, inverse_permutation(pr)]
        p_inv = inverse_permutation(p)
        del M, params
        # print(torch.norm(Y - X @ form_full(L, R, None, p, None)).item(), torch.norm(X_old @ M - X_old @ form_full(L, R, pl, p, pr)).item())
        errs.append((torch.norm(Y - X @ form_full(L, R, None, p, None)) / torch.norm(Y)).item())
        for i in range(approx_iters):
            C = torch.einsum('abc,bcd->abd', X.reshape(-1, kl, bl1), L).reshape(-1, kl * bl2)[:, p]
            C = rearrange(C, 'n (r b) -> r n b', r=kr)
            D = rearrange(Y, 'n (r b) -> r n b', r=kr)
            R = torch.linalg.lstsq(C, D).solution  # kr, br1, br2
            del C, D
            errs.append((torch.norm(Y - X @ form_full(L, R, None, p, None)) / torch.norm(Y)).item())
            PR = torch.block_diag(*[R[i] for i in range(len(R))])[p_inv, :].reshape(kl, bl2, -1)
            Qa, Ra = torch.linalg.qr(rearrange(X, 'a (k b) -> k a b', k=kl))  # kl, -1, bl1
            Qb, Rb = torch.linalg.qr(PR.transpose(1, 2))  # kl, -1, bl2
            a = min([torch.min(torch.abs(torch.diag(Ra[i]))) for i in range(Ra.shape[0])])
            b = min([torch.min(torch.abs(torch.diag(Rb[i]))) for i in range(Rb.shape[0])])
            if a < 1e-5 or b < 1e-5:
                del Qa, Qb, Ra, Rb, PR, a, b
                break

            def func(x0):
                return (torch.einsum('lma,lnb,lab->mn', Qa, Qb, x0.reshape(kl, bl1, bl2)) - Y).reshape(-1)

            res = torchmin.least_squares(func, L.reshape(-1), tr_solver='lsmr', ftol=1e-3, xtol=1e-3)
            L = res.x.reshape(kl, bl1, bl2)
            try:
                L = torch.linalg.solve_triangular(Ra, L, upper=True)
            except torch.linalg.LinAlgError:
                L = torch.bmm(torch.linalg.pinv(Ra, atol=1e-6), L)
            try:
                L = torch.linalg.solve_triangular(Rb.transpose(1, 2), L, upper=False, left=False)
            except torch.linalg.LinAlgError:
                L = torch.bmm(L, torch.linalg.pinv(Rb.transpose(1, 2), atol=1e-6))
            errs.append((torch.norm(Y - X @ form_full(L, R, None, p, None)) / torch.norm(Y)).item())
            del res, Qa, Qb, Ra, Rb, a, b
        print('monarch w', errs)
        del X, Y, errs, p_inv
        cleanup_memory()
        return L, R, pl, p, pr


def monarch_project_weighted(W: list, params, X: list, inp_x_out=True, inv=None, approx_iters=0, W_appr=None):
    if X is None:
        X = [None] * len(W)
    new_weights = []
    for i in range(len(W)):
        new_weights.append(project_matrix(W[i], params[i], X=X[i], inp_x_out=inp_x_out,
                                          approx_iters=approx_iters))
    del W, X
    cleanup_memory()
    return new_weights


def monarch_project(W, params):
    new_weights = []
    for i in range(len(W)):
        new_weights.append(project_matrix(W[i], params[i]))
    del W, params
    cleanup_memory()
    return new_weights


def monarch_merge(weights, params=None):
    if weights is None:
        return None
    new_weights = []
    for i in range(len(weights)):
        w = form_full(*weights[i])
        if params is not None:
            if 'diag' in params[i].keys():
                if w.shape[0] == params[i]['diag'].shape[0]:
                    w = w * params[i]['diag'][:, None]
                else:
                    w = w * params[i]['diag'][None, :]
        new_weights.append(w)
        del w
    del weights, params
    cleanup_memory()
    return new_weights
