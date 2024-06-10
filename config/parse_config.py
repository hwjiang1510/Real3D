
from easydict import EasyDict as edict

def parse_config(config):
    # model
    make_default(config, 'model.model_name', 'model.ckpt')
    # dataset
    make_default(config, 'dataset.normalization', 'constant_distance')
    assert config.dataset.normalization in ['constant_distance', 'constant_scale']
    make_default(config, 'dataset.sv_render_views', 0)
    make_default(config, 'dataset.sv_render_views_sample', 'none')
    make_default(config, 'dataset.sv_curriculum', 'none')
    make_default(config, 'dataset.sv_use_aug', False)
    make_default(config, 'dataset.omniobject3d_num_views', 10)
    make_default(config, 'dataset.co3d_num_views', 4)
    # train
    make_default(config, 'train.batch_size_sv', 0)
    make_default(config, 'train.use_zeroRO', False)
    make_default(config, 'train.use_consistency', False)
    make_default(config, 'train.num_frame_consistency', 0)
    make_default(config, 'train.rerender_consistency_input', False)

    if config.train.use_consistency:
        assert config.train.num_frame_consistency <= (config.dataset.sv_render_views - 1)
    return config


def make_default(config, key, default_val):
    # Split the key by '.' to navigate through the nested structure
    keys = key.split('.')
    # Start from the root of the config
    curr = config
    for k in keys[:-1]:  # Iterate through all but the last key
        # If the current key does not exist, create it as an empty EasyDict
        if not hasattr(curr, k):
            setattr(curr, k, edict())
        # Navigate deeper
        curr = getattr(curr, k)
    # Set the default value if the last key does not exist
    if not hasattr(curr, keys[-1]):
        setattr(curr, keys[-1], default_val)