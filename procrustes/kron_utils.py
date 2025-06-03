import torch
from einops import rearrange
from slicegpt.utils import cleanup_memory


@torch.no_grad()
def kron_project_matrix(M, params, X=None, XtX_inv=None, inp_x_out=True, approx_iters=0, A=None, B=None):
    m1, n1, r = params['m1'], params['n1'], params['r']
    m2, n2 = M.shape[0] // m1, M.shape[1] // n1
    if not inp_x_out:
        m2, n2 = M.shape[1] // m1, M.shape[0] // n1
    if X is None:
        # project M to A x_kron B: |M - A x_kron B|_F -> min
        assert M.shape[0] == m1 * m2 and M.shape[1] == n1 * n2

        left = True
        if 'diag' in params.keys():
            if M.shape[0] == params['diag'].shape[0]:
                M = M * params['diag'][:, None]
            else:
                left = False
                M = M * params['diag'][None, :]

        rearranged_M = rearrange(M.to(torch.float32), '(l i) (k j) -> (l k) (i j)', l=m1, k=n1)
        del M
        U, S, VT = torch.linalg.svd(rearranged_M, full_matrices=False)
        del rearranged_M

        Al = U[:, :r] * (S[:r]).sqrt()[None, :]
        Bl = (S[:r]).sqrt()[:, None] * VT[:r, :]
        del U, S, VT

        A = rearrange(Al, '(l k) r -> r l k', l=m1, r=r)
        B = rearrange(Bl, 'r (i j) -> r i j', i=m2, r=r)
        del Al, Bl
        cleanup_memory()

        if 'diag' in params.keys():
            if left:
                if A.shape[1] == params['diag'].shape[0]:
                    A = A / params['diag'][None, :, None]
                else:
                    B = B / params['diag'][None, :, None]
            else:
                if A.shape[2] == params['diag'].shape[0]:
                    A = A / params['diag'][None, None, :]
                else:
                    B = B / params['diag'][None, None, :]

        del params
        return A, B
    else:
        # project M to A x_kron B in weighted norm: |XM - X (A x_kron B)|_F -> min
        assert inp_x_out
        errs = []
        norm = torch.norm(X)
        # X = X / norm * int(X.shape[0] ** 0.25)  # for numerical stability
        if A is None or B is None or ((n1 == 1 and m1 == 1) or (n2 == 1 and m2 == 1)):
            A, B = kron_project_matrix(M, params, X=None)
        XtX = X.T @ X
        XtXM = (XtX @ M).reshape(m1, m2, n1, n2)
        flag = False
        XtX = XtX.reshape(m1, m2, m1, m2)
        M = (X @ M).reshape(-1, n1, n2)
        X = X.reshape(-1, m1, m2)
        errs.append(torch.norm(M - torch.einsum('sij,rik,rjn->skn', X, A, B)).item())
        if not ((n1 == 1 and m1 == 1) or (n2 == 1 and m2 == 1)) and not flag:
            for i in range(approx_iters):
                if m2 == 1:
                    if XtX_inv is None:
                        continue
                    # P = torch.linalg.inv(torch.einsum('rij,nkj->rn', B, B))
                    # Q = torch.einsum('rij,aibj->rab', B, XtXM)  # (r, m1, n1)
                    # A = torch.einsum('rn,ak,rab->nkb', P, XtX_inv, Q).reshape(r, m1, n1)
                    P = torch.linalg.inv(torch.einsum('rij,nkj->rn', B, B))
                    P = torch.einsum('rij,rn->nij', B, P)
                    P = torch.einsum('nij,aibj->nab', P, XtXM)  # (r, m1, n1)
                    A = torch.einsum('ak,nab->nkb', XtX_inv, P).reshape(r, m1, n1)
                else:
                    P = torch.einsum('rij,nkj,aibk->ranb', B, B, XtX).reshape(r * m1, r * m1)
                    Q = torch.einsum('rij,aibj->rab', B, XtXM).reshape(r * m1, n1)
                    try:
                        A = torch.linalg.solve(P, Q).reshape(r, m1, n1)
                    except torch.linalg.LinAlgError:
                        print('linalg error')
                        flag = True
                        break
                errs.append(torch.norm(M - torch.einsum('sij,rik,rjn->skn', X, A, B)).item())
                if m1 == 1:
                    if XtX_inv is None:
                        continue
                    # P = torch.linalg.inv(torch.einsum('rij,nkj->rn', A, A))
                    # Q = torch.einsum('rij,iajb->rab', A, XtXM)  # (r, m2, n2)
                    # B = torch.einsum('rn,ak,rab->nkb', P, XtX_inv, Q).reshape(r, m2, n2)
                    P = torch.linalg.inv(torch.einsum('rij,nkj->rn', A, A))
                    P = torch.einsum('rij,rn->nij', A, P)
                    P = torch.einsum('nij,iajb->nab', P, XtXM)  # (r, m2, n2)
                    B = torch.einsum('ak,nab->nkb', XtX_inv, P).reshape(r, m2, n2)
                else:
                    P = torch.einsum('rij,nkj,iakb->ranb', A, A, XtX).reshape(r * m2, r * m2)
                    Q = torch.einsum('rij,iajb->rab', A, XtXM).reshape(r * m2, n2)
                    try:
                        B = torch.linalg.solve(P, Q).reshape(r, m2, n2)
                    except torch.linalg.LinAlgError:
                        print('linalg error', P.shape)
                        flag = True
                        break

                errs.append(torch.norm(M - torch.einsum('sij,rik,rjn->skn', X, A, B)).item())
                del P, Q
                cleanup_memory()
        print('kron project', errs)
        del M, X, errs, XtXM, XtX, XtX_inv, params
        cleanup_memory()
        return A.to(torch.float32), B.to(torch.float32)


def kron_project_weighted(W: list, params, X: list, inv: list, inp_x_out=None, approx_iters=0, W_appr=None):
    if X is None:
        X = [None] * len(W)
        inv = [None] * len(W)
    new_weights = []
    for i in range(len(W)):
        A, B = None, None
        if W_appr is not None:
            A, B = W_appr[i][0], W_appr[i][1]
        new_weights.append(kron_project_matrix(W[i], params[i], X=X[i], XtX_inv=inv[i], inp_x_out=inp_x_out,
                                               approx_iters=approx_iters, A=A, B=B))
    del W, X, W_appr, A, B
    cleanup_memory()
    return new_weights


def kron_project(W, params):
    new_weights = []
    for i in range(len(W)):
        new_weights.append(kron_project_matrix(W[i], params[i]))
    del W, params
    cleanup_memory()
    return new_weights


def kron_merge(weights, params=None):
    if weights is None:
        return None
    new_weights = []
    for i in range(len(weights)):
        r, m1, n1 = weights[i][0].shape
        _, m2, n2 = weights[i][1].shape
        w = torch.einsum('iab,icd->acbd', weights[i][0], weights[i][1]).reshape((m1 * m2, n1 * n2))
        if params is not None:
            if 'diag' in params[i].keys():
                if w.shape[0] == params[i]['diag'].shape[0]:
                    w = w * params[i]['diag'][:, None]
                else:
                    w = w * params[i]['diag'][None, :]
        new_weights.append(w)
    del weights, params
    cleanup_memory()
    return new_weights
