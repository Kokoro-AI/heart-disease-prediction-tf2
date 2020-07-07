from .heart import load_heart

def load(data_dir, config, use_feature_transform=False, numeric=False, categorical=False):
    """
    Load specific dataset.
    Args:
        data_dir (str): path to the dataset directory.
        config (dict): general dict with settings.
        use_feature_transform (bool): apply dense feature transform or not
    Returns (dict): tensorflow Dataset objects and features.
    """

    if config['data.dataset'] == "heart":
        ret = load_heart(data_dir, config,
                         use_feature_transform=use_feature_transform,
                         numeric=numeric, categorical=categorical)
    else:
        raise ValueError(f"Unknow dataset: {config['data.dataset']}")

    return ret
