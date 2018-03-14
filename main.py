import os.path
import tensorflow as tf
import helper
import warnings
import argparse
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer. ' \
                                                            ' You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, l3_out, l4_out, l7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    weight_scale = 1e-3
    reg = tf.contrib.layers.l2_regularizer(scale=weight_scale)
    # Apply 1x1 convolution to layer 7 output to produce layer 7 class predictions
    l7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, (1,1), padding='same', kernel_regularizer=reg)
    # Upsample layer 7 class predictions by 4x
    l4_part1 = tf.layers.conv2d_transpose(l7_1x1, num_classes, 4, 2, padding='same', kernel_regularizer=reg)
    # Apply 1x1 convolution to layer 4 output to produce layer 4 class predictions
    l4_part2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, (1,1), padding='same', kernel_regularizer=reg)
    l4_part2_scaled = tf.multiply(l4_part2, 0.01, name='pool4_scaled')
    # Apply skip connections to layer 4
    l4_merged = tf.add(l4_part1, l4_part2_scaled)
    # Upsample the merged layer 4
    l3_part1 = tf.layers.conv2d_transpose(l4_merged, num_classes, 4, 2, padding='same', kernel_regularizer=reg)
    # Apply 1x1 convolution to layer 3 output to produce layer 3 class predictions
    l3_part2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, (1, 1), padding='same', kernel_regularizer=reg)
    l3_part2_scaled = tf.multiply(l3_part2, 0.0001, name='pool3_scaled')
    # Apply skip connection to layer 3
    l3_merged = tf.add(l3_part1, l3_part2_scaled)
    # Upsample the merged layer 3
    output = tf.layers.conv2d_transpose(l3_merged, num_classes, 16, 8, padding='same', kernel_regularizer=reg)

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    l2_loss = tf.losses.get_regularization_loss()
    train_op = optimizer.minimize(loss_op + l2_loss)

    return logits, train_op, loss_op


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    lrate = 5e-4
    kp = 0.5

    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        print("Epoch: ", i)
        for image, mask in get_batches_fn(batch_size):
            loss = sess.run([train_op, cross_entropy_loss],
                            feed_dict={input_image: image, correct_label: mask,
                                       learning_rate: lrate, keep_prob: kp})
        print("Training loss: ", loss[1])


tests.test_train_nn(train_nn)


def run():
    # Define data paths
    data_dir = os.path.join(args.data_path)
    runs_dir = os.path.join('.', 'runs')

    # Define constants
    num_classes = 2
    image_shape = (160, 576)
    batch_size = args.batch_size
    epochs = args.epochs

    # Declare tf variables
    learning_rate = tf.placeholder(tf.float32)
    gt_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road', 'training'), image_shape)
        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)
        output_layer = layers(l3_out, l4_out, l7_out, num_classes)
        logits, train_op, loss = optimize(output_layer, gt_label, learning_rate, num_classes)
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, input_image, gt_label, keep_prob,
                 learning_rate)
        # Save inference samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation using VGG16-FCN8')
    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        default='d:\\',
        help='Path to training data'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=10,
        help='# of epochs to train the model'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=2,
        help='# of samples in batch size'
    )

    args = parser.parse_args()
    run()
