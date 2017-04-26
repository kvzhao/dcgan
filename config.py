class ModelConfig(object):
    task_name = 'test-dcgan'

    batch_size = 64
    sample_num = 64
    train_size = 100000

    input_height = 32
    input_width = 32
    output_height = 32
    output_width= 32

    z_dim = 100
    gf_dim = 64
    df_dim = 64
    gfc_dim = 1024
    dfc_dim = 1024
    c_dim = 1 # dimension of channel
    is_gray = (c_dim == 1)

    '''
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
      '''

class SolverConfig(object):
    learning_rate = 0.0001
    beta1 = 0.5
    epoch = 2000

    save_per_iter = 1000
    sample_per_iter = 500
    display_status_per_iter = 20

class DCGANConfig(ModelConfig, SolverConfig):
    pass

class M(DCGANConfig):
    backend = 'tf'

def get_config(FLAGS):
    config = M
    for k, v in FLAGS.__dict__['__flags'].items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config
