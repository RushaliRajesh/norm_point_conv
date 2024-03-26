import os
import logging
import tensorflow as tf
import numpy as np

# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set TensorFlow logging verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable TensorFlow logging
logging.getLogger('tensorflow').disabled = True

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device("/cpu:0"):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    # with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            net = conv2d(net, num_hidden_units, [1, 1],
                                padding = 'VALID', stride=[1, 1],
                                bn = True, is_training = is_training, activation_fn=activation_fn,
                                scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

            #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp%d'%(i))
        return net


if __name__ == '__main__':
    
    # xyz = tf.random.uniform((3, 100, 3))
    # features = tf.random.uniform((3, 100, 6))
    # npoint = 50
    # nsample = 32
    # mlp = [32, 32, 64]
    # new_xyz = tf.random.uniform((3, npoint, 3))
    # grouped_xyz = tf.random.uniform((3, npoint, nsample, 3))
    # grouped_features = tf.random.uniform((3, npoint, nsample, 6))
    # density_scale = tf.random.uniform((3, npoint, nsample, 1))
    # weight = weight_net_hidden(grouped_xyz, [32], scope = 'decode_weight_net', is_training=True)
   
    print("testing: ")
    #create a random tensor tensorflow
    xyz = tf.random.uniform((2, 10, 5,3))
    weight = tf.random.uniform((2, 10, 5, 32)) #after weightnet
    print(xyz.shape)
    #a tf.nn conv2d layer with 32 out dim               
    kernel_size = (1, 1)  # Assuming a 3x3 kernel size

    # Define the number of output channels
    num_output_channels = 32

    # Create the weight tensor (kernel) for the convolutional layer
    kernel = tf.Variable(tf.random.normal([kernel_size[0], kernel_size[1], xyz.shape[-1], num_output_channels]))

    # Perform the convolution operation
    out = tf.nn.conv2d(input=xyz, filters=kernel, strides=[1, 1, 1, 1], padding='VALID')

    # Print the shape of the output tensor
    print(out.shape)
    grouped_feature = tf.random.uniform((2, 10, 5, 32))
    density_scale = tf.random.uniform((2, 10, 5, 1))
    new_points = tf.multiply(grouped_feature, density_scale)
    print("multiply (grouped_features, density_scales) (= new_points)",new_points.shape)
    new_points = tf.transpose(new_points, [0, 1, 3, 2])
    print("transpose new_points:",new_points.shape)
    new_points = tf.matmul(new_points, weight)
    print("matmul new_points and weight:",new_points.shape)
    mlp = [32,32,64]

    kernel_2 = tf.Variable(tf.random.normal([1, new_points.shape[2], new_points.shape[-1], mlp[-1]]))
    print()
    print("last cnn")
    print("kernel_2:",kernel_2.shape)
    print("new_points:",new_points.shape)
    new_points = tf.nn.conv2d(input=new_points, filters=kernel_2, strides=[1, 1, 1, 1], padding='VALID')
    print("conv2d new_points:",new_points.shape)
       
