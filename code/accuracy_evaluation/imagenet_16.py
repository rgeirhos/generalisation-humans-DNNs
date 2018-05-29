"""
Functionality to run ImageNet images through a model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.resnet import resnet_run_loop
from official.resnet import imagenet_main
from official.resnet import resnet_model


_NUM_IMAGES = {
    # approx based on batch size 64
    # (last batch might have been smaller)
    'train': 3581 * 64,
}


class Imagenet16Model(imagenet_main.ImagenetModel):
    def __init__(self, resnet_size, data_format=None, num_classes=16,
                 version=resnet_model.DEFAULT_VERSION):
        return super(Imagenet16Model, self).__init__(
            resnet_size=resnet_size,
            data_format=data_format,
            num_classes=num_classes,
            version=version)


def imagenet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

    return resnet_model_fn(features, labels, mode,
                           Imagenet16Model,
                           resnet_size=params['resnet_size'],
                           weight_decay=1e-4,
                           learning_rate_fn=learning_rate_fn,
                           momentum=0.9,
                           data_format=params['data_format'],
                           version=params['version'],
                           loss_filter_fn=None,
                           multi_gpu=params['multi_gpu'])


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, version, loss_filter_fn=None,
                    multi_gpu=False):

    weights = features['weight']
    features = features['image']

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)

    model = model_class(resnet_size, data_format, version=version)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'logits': logits,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2
    # regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels, weights=weights,
        reduction=tf.losses.Reduction.MEAN)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    if not loss_filter_fn:
        def loss_filter_fn(name):
            return 'batch_normalization' not in name

    # Add weight decay to the loss.
    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if multi_gpu:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    raw_accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'], weights=weights)
    mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(
        tf.argmax(labels, axis=1), predictions['classes'], 16, weights=weights)

    metrics = {
        'raw_accuracy': raw_accuracy,
        'accuracy': accuracy,
        'mean_per_class_accuracy': mean_per_class_accuracy,
    }

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(raw_accuracy[1], name='raw_train_accuracy')
    tf.summary.scalar('raw_train_accuracy', raw_accuracy[1])
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.identity(mean_per_class_accuracy[1],
                name='mean_per_class_train_accuracy')
    tf.summary.scalar('mean_per_class_train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)
