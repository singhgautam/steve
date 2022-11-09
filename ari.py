import torch

from scipy.special import comb


def compute_ari(table):
    """
    Compute ari, given the index table
    :param table: (r, s)
    :return:
    """

    # (r,)
    a = table.sum(axis=1)
    # (s,)
    b = table.sum(axis=0)
    n = a.sum()

    comb_a = comb(a, 2).sum()
    comb_b = comb(b, 2).sum()
    comb_n = comb(n, 2)
    comb_table = comb(table, 2).sum()

    if (comb_b == comb_a == comb_n == comb_table):
        # the perfect case
        ari = 1.0
    else:
        ari = (
                (comb_table - comb_a * comb_b / comb_n) /
                (0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n)
        )

    return ari


def compute_mask_ari(mask0, mask1):
    """
    Given two sets of masks, compute ari
    :param mask0: ground truth mask, (N0, D)
    :param mask1: predicted mask, (N1, D)
    :return:
    """

    # will first need to compute a table of shape (N0, N1)
    # (N0, 1, D)
    mask0 = mask0[:, None].byte()
    # (1, N1, D)
    mask1 = mask1[None, :].byte()
    # (N0, N1, D)
    agree = mask0 & mask1
    # (N0, N1)
    table = agree.sum(dim=-1)

    return compute_ari(table.numpy())


def evaluate_ari(true_mask, pred_mask):
    """
    :param
        true_mask: (B, N0, D)
        pred_mask: (B, N1, D)
    :return: average ari
    """
    from torch import arange as ar

    B, K, D = pred_mask.size()

    # max_index (B, D)
    max_index = torch.argmax(pred_mask, dim=1)

    # get binarized masks (B, N1, D)
    pred_mask = torch.zeros_like(pred_mask)
    pred_mask[ar(B)[:, None], max_index, ar(D)[None, :]] = 1.0

    aris = 0.
    for b in range(B):
        aris += compute_mask_ari(true_mask[b].detach().cpu(), pred_mask[b].detach().cpu())

    avg_ari = aris / B
    return avg_ari
