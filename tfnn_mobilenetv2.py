import tensorflow as tf
# redefine mobilnet funcs in pure tf

random_normal = tf.initializers.RandomNormal()

def correct_pad(inputs, kernel_size):
    # channels last
    img_dim = 1
    input_size = tf.shape(inputs)[img_dim:(img_dim + 2)].numpy()

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))
    
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def pad2d(x, pad=(0, 0, 0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]]
    x = tf.pad(x, paddings, mode=mode, constant_values=0)
    return x

# inverted residual blocks with tf nn
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    # for now assume weight and bias are the same
    # take care of params
    in_channels = tf.shape(inputs)[-1].numpy()
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)
    if block_id:
      # expansion
      x = tf.nn.conv2d(x,
                       tf.Variable(random_normal([1, 1, in_channels, (expansion*in_channels)])),
                       strides=[1, 1, 1, 1], padding='SAME',
                       name=prefix+'expand')
      # calc mean and var
      x_mean, x_std = tf.nn.moments(x, axes = 2, keepdims=True)
      x = tf.nn.batch_normalization(x, x_mean, x_std, None, None, 1e-12,
                                    name=prefix+'expand_BN')
      x = tf.nn.relu6(x, name=prefix+'expand_relu')
      print(f'[INFO] shape after expansion: \n{x.shape}\n')
    else:
      prefix = 'expanded_conv_'
    
    # Depthwise
    if stride == 2:
        padding = correct_pad(x, 3)
        x = pad2d(x, pad=(padding[0][0], padding[0][1], padding[1][0], padding[1][1]))
        print(f'[INFO] shape after padding2d: \n{x.shape}\n')
    x = tf.nn.depthwise_conv2d(x, tf.Variable(random_normal([3, 3, tf.shape(x)[-1].numpy(), 1])),
                               strides=[1, stride, stride, 1],
                               padding='SAME' if stride == 1 else 'VALID',
                               name=prefix+'depthwise')
    x_mean, x_std = tf.nn.moments(x, axes = 2, keepdims=True)
    x = tf.nn.batch_normalization(x, x_mean, x_std, None, None, 1e-12,
                                  name=prefix+'depthwise_BN')
    x = tf.nn.relu6(x, name=prefix+'depthwise_relu')
    print(f'[INFO] shape after depthwiseconv: \n{x.shape}\n')
    # projection
    x = tf.nn.conv2d(x,
                     tf.Variable(random_normal([1, 1, tf.shape(x)[-1].numpy(), pointwise_filters])),
                     strides=[1, 1, 1, 1], padding='SAME',
                     name=prefix+'project')
    # calc mean and var
    x_mean, x_std = tf.nn.moments(x, axes = 2, keepdims=True)
    x = tf.nn.batch_normalization(x, x_mean, x_std, None, None, 1e-12,
                                  name=prefix + 'project_BN')
    
    # add
    if in_channels == pointwise_filters and stride == 1:
      x = tf.add(inputs, x,
                 name=prefix+'add')

    print(f'[INFO] shape end of the layer: \n{x.shape}\n')
    return x

def mobilenetv2(x, alpha=1.0, classes=6):
    first_block_filters = _make_divisible(32 * alpha, 8)
    in_channels = tf.shape(x)[-1].numpy()
    padding = correct_pad(inputs=x, kernel_size=3)

    # starting layers
    x = pad2d(x, pad=(padding[0][0], padding[0][1], padding[1][0], padding[1][1]))
    x = tf.nn.conv2d(x,
                     tf.Variable(random_normal([3, 3, in_channels, first_block_filters])),
                     strides=[1, 2, 2, 1], padding='VALID',
                     name="Conv1")
    x_mean, x_std = tf.nn.moments(x, axes = 2, keepdims=True)
    x = tf.nn.batch_normalization(x, x_mean, x_std, None, None, 1e-12,
                                  name='bn_conv1')
    x = tf.nn.relu6(x, name="Conv1_relu")

    # the mobilenet architecture
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = tf.nn.conv2d(x,
                     tf.Variable(random_normal([1, 1, tf.shape(x)[-1].numpy(), last_block_filters])),
                     strides=[1, 1, 1, 1], padding='VALID',
                     name="Conv_out")
    x_mean, x_std = tf.nn.moments(x, axes = 2, keepdims=True)
    x = tf.nn.batch_normalization(x, x_mean, x_std, None, None, 1e-12,
                                  name='out_bn')
    x = tf.nn.relu6(x, name="out_relu")

    # global average pool
    x = tf.nn.avg_pool2d(x, ksize=7, strides=1, padding='VALID')
    x = tf.reshape(x, [-1, last_block_filters])
    print(f'[INFO] global average pool: {x.shape}')
    # last dense
    x = tf.matmul(x,
                  tf.Variable(random_normal([last_block_filters, classes])))
    # labels
    x = tf.nn.softmax(x)
    print(f'[INFO] output shape: {x.shape}')
    return x

import numpy as np

inputs = tf.cast(np.random.rand(1, 224, 224, 1), dtype=tf.float32)

output = mobilenetv2(inputs, 1.0, 6)
print(output.shape)
