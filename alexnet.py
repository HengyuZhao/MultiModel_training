# copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Timing benchmark for AlexNet inference.

To run, use:
  bazel run -c opt --config=cuda \
      models/tutorials/image/alexnet:alexnet_benchmark

Across 100 steps on batch size = 128.

Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch

Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time
from PIL import Image
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline

FLAGS = None


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images, return_layers=False):
  """Build the AlexNet model.

  Args:
    images: Images Tensor

  Returns:
    pool5: the last Tensor in the convolutional component of AlexNet.
    parameters: a list of Tensors corresponding to the weights and biases of the
        AlexNet model.
  """
  layers = dict()

  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]
    layers["conv1"] = conv1
  # lrn1
 # with tf.name_scope('lrn1') as scope:
  #  lrn1 = tf.nn.local_response_normalization(conv1,
  #                                            alpha=1e-4,
  #                                            beta=0.75,
   #                                           depth_radius=2,
   #                                           bias=2.0)

  # pool1
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
  print_activations(pool1)
  layers["pool1"] = pool1

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(conv2)
  layers["conv2"] = conv2
  # lrn2
 # with tf.name_scope('lrn2') as scope:
 #   lrn2 = tf.nn.local_response_normalization(conv2,
  #                                            alpha=1e-4,
  #                                            beta=0.75,
   #                                           depth_radius=2,
   #                                           bias=2.0)

  # pool2
  pool2 = tf.nn.max_pool(conv2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  print_activations(pool2)
  layers["pool2"] = pool2

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)
  layers["conv3"] = conv3

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)
  layers["conv4"]  = conv4

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)
  layers["conv5"] = conv5

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  print_activations(pool5)
  layers["pool5"] = pool5

  # dense1
  dense1 = tf.layers.dense(inputs=pool5, units=4096, activation=tf.nn.relu)
  layers["dense1"] = dense1

  # dense2
  dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)
  layers["dense2"] = dense2
  
  # dense3
  dense3 = tf.layers.dense(inputs=dense2, units=1000, activation=tf.nn.relu)
  layers["dense3"] = dense3
  if return_layers:
  	return dense3, parameters, layers
  return dense3, parameters


def image_smys(layers):
  smy_writer = tf.summary.FileWriter("tensorboard_log", graph=tf.get_default_graph())
  saver = tf.train.Saver()
  by_channels = tf.unstack(layers["conv5"], axis=3)
  for idx, c in enumerate(by_channels):
    print(tf.expand_dims(c, axis=-1))
    tf.summary.image("conv5_channel_" + str(idx), tf.expand_dims(c, axis=-1))
  return tf.summary.merge_all(), smy_writer, saver


def time_tensorflow_run(session, target, info_string, smy=None, writer=None, saver=None):
  """Run the computation to obtain the target tensor and print timing stats.

  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.

  Returns:
    None
  """
  if smy is not None:
    assert writer is not None, "a writer must be provided to log images"
    assert saver is not None, "a saver must be provided to log images"
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    if smy is None:
      _ = session.run(target,options=options,run_metadata=run_metadata)
    else:
      _, feature_map = session.run([target, smy],options=options,run_metadata=run_metadata)
      writer.add_summary(feature_map)
      saver.save(session, "tensorboard_log/alexnet.ckpt")
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
   # with open('/home/hzhao28/PIM/alexnet.json', 'w') as f:
     #   f.write(chrome_trace)
   # print("saved")
   # raise Exception
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))



def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    # images = tf.Variable(tf.random_normal([FLAGS.batch_size,
    #                                        image_size,
    #                                        image_size, 3],
    #                                       dtype=tf.float32,
    #                                       stddev=1e-1))

    image = Image.open("Picture1.png").resize((image_size, image_size))
    image_rgb = Image.new("RGB", image.size, (255, 255, 255))
    image_rgb.paste(image, mask=image.split()[3])
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    images = np.repeat(np.reshape(image_rgb, [1, image_size, image_size, 3]), FLAGS.batch_size, axis=0)
    

    # Build a Graph that computes the logits predictions from the
    # inference model.
    # dense3, parameters = inference(images)
    dense3, parameters, layers = inference(images, True)
    
    feature_maps = image_smys(layers)

    # Add a simple objective so we can calculate the backward pass.
    objective = tf.nn.l2_loss(dense3)
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, parameters)

    # this is not forward-backward time. tf.gradient only computes gradient wrt trainable variables
    # to actucally do back propagation, try this:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    grad = optimizer.compute_gradients(objective)
    bp_op = optimizer.apply_gradients(grad)

  # or if gradients themselve is not of interest
    bp_op = optimizer.minimize(objective)

    # Build an initialization operation.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth=True
   # config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(init)
   # print(sess.run(dense3))
    # Run the forward benchmark.
    time_tensorflow_run(sess, dense3, "Forward", *feature_maps)

    time_tensorflow_run(sess, layers["conv5"], "5th layer feature map - Forward", *feature_maps)

    time_tensorflow_run(sess, bp_op, "Forward-backward", *feature_maps)
    


    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")


def main(_):
  test_time = time.time()
  run_benchmark()
  print ("Terminal time: %4.4f" % (time.time() - test_time))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch size.'
  )
  parser.add_argument(
      '--num_batches',
      type=int,
      default=1000,
      help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
