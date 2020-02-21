from .heart import load_heart

def load(data_dir, config, splits):
    """
    Load specific dataset.
    Args:
        data_dir (str): path to the dataset directory.
        config (dict): general dict with settings.
        splits (list): list of strings 'train'|'val'|'test'.
    Returns (dict): dictionary with keys 'train'|'val'|'test'| and values
    as tensorflow Dataset objects and features.
    """

    if config['data.dataset'] == "heart":
        ret = load_heart(data_dir, config, splits)
    else:
        raise ValueError(f"Unknow dataset: {config['data.dataset']}")

    return ret