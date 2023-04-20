import numpy as np

def filter_cells(x, threshold=0, return_ids=False):
    """filter out cells with total counts less than or equal to thereshold

    Args:
        x (_type_): (genes x cells)
        threshold (int, optional): _description_. Defaults to 0.
        return_ids (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    ids = np.where(x.sum(axis=0) > threshold)[0]
    if return_ids: return x[:, ids], ids
    else: return x[:, ids]

def filter_genes(x, threshold=0, return_ids=False):
    """filter out genes with total counts less than or equal to thereshold

    Args:
        x (_type_): (genes x cells)
        threshold (int, optional): _description_. Defaults to 0.
        return_ids (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    ids = np.where(x.sum(axis=1) > threshold)[0]
    if return_ids: return x[ids, :], ids
    else: return x[ids, :]

def lib_size_normalize(x, to_size=10000, return_scale=False):
    """library size normalization.
    need to do this to prevent the bernoulli network giving nans from log(sigmoid)

    Args:
        x (_type_): (genes x cells)
        to_size (int, optional): the target library size. Defaults to 1.

    Returns:
        _type_: _description_
    """
    sizes = x.sum(axis=1)[:, None]
    scale = sizes / to_size
    return x / scale, scale

def log_transform(x, add=1.):
    return np.log(x + add)

