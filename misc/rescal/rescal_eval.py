import numpy as np
from numpy import dot, zeros
from numpy.linalg import norm
from sklearn.metrics import precision_recall_curve, auc
from rescal import als

def predict_rescal_als(T):
    A, R, _, _, _ = als(T, 50, init='nvecs', conv=1e-3, lambda_A=10, lambda_R=10, maxIter=20)
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P

def normalize_predictions(P, e, k):
    for a in range(e):
        for b in range(e):
            nrm = norm(P[a, b, :k])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P

def innerfold(T, mask_idx, target_idx, e, k, sz, GROUND_TRUTH):
    Tc = [Ti.copy() for Ti in T]
    mask_idx = np.unravel_index(mask_idx, (e, e, k))
    target_idx = np.unravel_index(target_idx, (e, e, k))

    # set values to be predicted to zero
    for i in range(len(mask_idx[0])):
        Tc[mask_idx[2][i]][mask_idx[0][i], mask_idx[1][i]] = 0

    # predict unknown values
    P = predict_rescal_als(Tc)
    P = normalize_predictions(P, e, k)

    # compute area under precision recall curve
    prec, recall, _ = precision_recall_curve(GROUND_TRUTH[target_idx], P[target_idx])
    return auc(recall, prec)