import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import tensorflow as tf


def privacy_custom_new():
    l2_norm_clip = 1.5
    noise_multiplier = 1.3
    num_microbatches = 250
    learning_rate = 0.25
    batch_size = 500

    if batch_size % num_microbatches != 0:
        raise ValueError('Batch size should be an integer multiple of the number of microbatches')

    optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate)

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)