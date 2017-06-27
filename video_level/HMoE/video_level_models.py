# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
from tensorflow import nn
import tensorflow.contrib.slim as slim

import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

flags.DEFINE_integer(
    "coarse_num_mixtures", 8,
    "The number of coarse classification mixtures (excluding the dummy 'expert') used for HMoeModel.")
flags.DEFINE_integer(
    "label_num_mixtures", 8,
    "The number of label classification mixtures (excluding the dummy 'expert') used for HMoeModel.")
flags.DEFINE_integer(
    "rank_of_basis", 512,
    "The rank of the basis for the probability space")

flags.DEFINE_string(
    "refine", "/mnt/lustre/share/dengby/video-pred/stats_new.npy",
    "The file recording stats of training set.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class HMLPModel(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   is_training=True,
                   input_size=1024 + 128,
                   **unused_params):
    """Creates a Multi Layered Perceptron model.
    """
    # General transform layer
    fc1 = slim.fully_connected(
        model_input,
        input_size,
        activation_fn = None,
        weights_regularizer = slim.l2_regularizer(l2_penalty),
        scope = 'fc1')
    bn1 = slim.batch_norm(fc1, is_training = is_training, scope = 'bn1')
    relu1 = nn.relu(bn1, name = 'relu1')

    # Coarse classification
    coarse_scores = slim.fully_connected(
        relu1,
        25,
        activation_fn = None,
        weights_regularizer = slim.l2_regularizer(l2_penalty),
        scope = 'coarse')

    # Concatenate p(coarse) and prior features
    concat = tf.concat([relu1, coarse_scores], -1, name = 'concat')

    # Specific transform layer
    fc2 = slim.fully_connected(
        concat,
        input_size + 25,
        activation_fn = None,
        weights_regularizer = slim.l2_regularizer(l2_penalty),
        scope = 'fc2')
    bn2 = slim.batch_norm(fc2, is_training = is_training, scope = 'bn2')
    relu2 = nn.relu(bn2, name = 'relu2')

    # Final classifier
    classifier = slim.fully_connected(
        relu2,
        vocab_size,
        activation_fn = None,
        weights_regularizer = slim.l2_regularizer(l2_penalty),
        scope = 'classifier')

    final_probs = nn.sigmoid(classifier, name = 'final_probs')
    coarse_probs = nn.sigmoid(coarse_scores, name = 'coarse_probs')

    return {"predictions": final_probs, "coarse_predictions": coarse_probs}

class HMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   coarse_num_mixtures=None,
                   label_num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   phase=True,
                   **unused_params):
    coarse_num_mixtures = coarse_num_mixtures or FLAGS.coarse_num_mixtures
    label_num_mixtures = label_num_mixtures or FLAGS.label_num_mixtures

    ### Coarse Level
    coarse_gate_activations = slim.fully_connected(
        model_input,
        25 * (coarse_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="coarse_gates")
    coarse_expert_activations = slim.fully_connected(
        model_input,
        25 * coarse_num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="coarse_experts")

    coarse_gating_distribution = tf.nn.softmax(tf.reshape(
        coarse_gate_activations,
        [-1, coarse_num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    coarse_expert_distribution = tf.nn.sigmoid(tf.reshape(
        coarse_expert_activations,
        [-1, coarse_num_mixtures]))  # (Batch * #Labels) x num_mixtures


    coarse_probabilities_by_class_and_batch = tf.reduce_sum(
        coarse_gating_distribution[:, :coarse_num_mixtures] * coarse_expert_distribution, 1)
        # coarse_survived_experts[:, :coarse_num_mixtures] * coarse_expert_distribution, 1)
    coarse_probabilities = tf.reshape(coarse_probabilities_by_class_and_batch,
                                     [-1, 25])

    concat = tf.concat([model_input, coarse_probabilities], -1, name = 'middle_concat')

    ### Label Level
    label_gate_activations = slim.fully_connected(
        concat,
        vocab_size * (label_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_gates")
    label_expert_activations = slim.fully_connected(
        concat,
        vocab_size * label_num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_experts")

    label_gating_distribution = tf.nn.softmax(tf.reshape(
        label_gate_activations,
        [-1, label_num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    label_expert_distribution = tf.nn.sigmoid(tf.reshape(
        label_expert_activations,
        [-1, label_num_mixtures]))  # (Batch * #Labels) x num_mixtures

    label_probabilities_by_class_and_batch = tf.reduce_sum(
        label_gating_distribution[:, :label_num_mixtures] * label_expert_distribution, 1)
        # survived_experts[:, :label_num_mixtures] * label_expert_distribution, 1)
    label_probabilities = tf.reshape(label_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    return {"predictions": label_probabilities, "coarse_predictions": coarse_probabilities}

class ColRelModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   coarse_num_mixtures=None,
                   label_num_mixtures=None,
                   rank_of_basis=None,
                   l2_penalty=1e-8,
                   phase=True,
                   **unused_params):
    coarse_num_mixtures = coarse_num_mixtures or FLAGS.coarse_num_mixtures
    label_num_mixtures = label_num_mixtures or FLAGS.label_num_mixtures
    rank_of_basis = rank_of_basis or FLAGS.rank_of_basis

    ### Coarse Level
    coarse_gate_activations = slim.fully_connected(
        model_input,
        25 * (coarse_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="coarse_gates")
    coarse_expert_activations = slim.fully_connected(
        model_input,
        25 * coarse_num_mixtures,
        # activation_fn=nn.relu,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="coarse_experts")

    coarse_gating_distribution = tf.nn.softmax(tf.reshape(
        coarse_gate_activations,
        [-1, coarse_num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    # coarse_expert_distribution = tf.nn.sigmoid(tf.reshape(
    coarse_expert_distribution = tf.reshape(
        coarse_expert_activations,
        # [-1, coarse_num_mixtures]))  # (Batch * #Labels) x num_mixtures
        [-1, 25 * coarse_num_mixtures])  # Batch x (#Labels * num_mixtures)

    coarse_normed_indie = tf.reshape(slim.batch_norm(
        coarse_expert_distribution,
        center=True,
        scale=True,
        activation_fn=nn.relu,
        is_training=phase,
        scope="coarse_bn"),
        [-1, coarse_num_mixtures])

    coarse_probabilities_by_class_and_batch = tf.reduce_sum(
        # coarse_gating_distribution[:, :coarse_num_mixtures] * coarse_expert_distribution, 1)
        coarse_gating_distribution[:, :coarse_num_mixtures] * coarse_normed_indie, 1)
    coarse_indie_probabilities = tf.reshape(coarse_probabilities_by_class_and_batch,
                                     [-1, 25])

    coarse_scores = slim.fully_connected(
        coarse_indie_probabilities,
        25,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="actual_coarse")
    coarse_probabilities = nn.sigmoid(coarse_scores)

    concat = tf.concat([model_input, nn.relu(coarse_scores)], -1, name = 'middle_concat')

    ### Label Level
    label_gate_activations = slim.fully_connected(
        concat,
        vocab_size * (label_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_gates")
    label_expert_activations = slim.fully_connected(
        concat,
        vocab_size * label_num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_experts")

    label_gating_distribution = tf.nn.softmax(tf.reshape(
        label_gate_activations,
        [-1, label_num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    # label_expert_distribution = tf.nn.sigmoid(tf.reshape(
    label_expert_distribution = tf.reshape(
        label_expert_activations,
        # [-1, label_num_mixtures]))  # (Batch * #Labels) x num_mixtures
        [-1, vocab_size * label_num_mixtures])  # Batch x (#Labels * num_mixtures)

    label_normed_indie = tf.reshape(slim.batch_norm(
        label_expert_distribution,
        center=True,
        scale=True,
        activation_fn=nn.relu,
        is_training=phase,
        scope="label_bn"),
        [-1, label_num_mixtures])

    label_probabilities_by_class_and_batch = tf.reduce_sum(
        # label_gating_distribution[:, :label_num_mixtures] * label_expert_distribution, 1)
        label_gating_distribution[:, :label_num_mixtures] * label_normed_indie, 1)
        # label_normed_indie[:, :label_num_mixtures] * label_expert_distribution, 1)
    label_indie_probabilities = tf.reshape(label_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    projection_to_basis = slim.fully_connected(
        label_indie_probabilities,
        rank_of_basis,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="reduce_to_basis")
    reduce_normed_indie = slim.batch_norm(
        projection_to_basis,
        center=True,
        scale=True,
        activation_fn=nn.relu,
        is_training=phase,
        scope="reduce_bn")

    reduce_propogation = slim.fully_connected(
        # projection_to_basis,
        reduce_normed_indie,
        rank_of_basis,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="propogation")
    transformed_normed_indie = slim.batch_norm(
        reduce_propogation,
        center=True,
        scale=True,
        activation_fn=nn.relu,
        is_training=phase,
        scope="transform_bn")
    
    label_scores = slim.fully_connected(
        # reduce_propogation,
        transformed_normed_indie,
        vocab_size,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_prob")
    label_probabilities = nn.sigmoid(label_scores)

    return {"predictions": label_probabilities, "coarse_predictions": coarse_probabilities}

class RefineModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   coarse_num_mixtures=None,
                   label_num_mixtures=None,
                   l2_penalty=1e-8,
                   phase=True,
                   **unused_params):
    coarse_num_mixtures = coarse_num_mixtures or FLAGS.coarse_num_mixtures
    label_num_mixtures = label_num_mixtures or FLAGS.label_num_mixtures

    ### Coarse Level
    coarse_gate_activations = slim.fully_connected(
        model_input,
        25 * (coarse_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="coarse_gates")
    coarse_expert_activations = slim.fully_connected(
        model_input,
        25 * coarse_num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="coarse_experts")

    coarse_gating_distribution = tf.nn.softmax(tf.reshape(
        coarse_gate_activations,
        [-1, coarse_num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    coarse_expert_distribution = tf.nn.sigmoid(tf.reshape(
        coarse_expert_activations,
        [-1, coarse_num_mixtures]))  # (Batch * #Labels) x num_mixtures

    ### Randomly sick
    coarse_survived_experts = slim.dropout(
        coarse_gating_distribution,
        keep_prob=0.75,
        noise_shape=tf.constant([1, coarse_num_mixtures + 1]),
        is_training=phase,
        scope="coarse_drop")

    coarse_probabilities_by_class_and_batch = tf.reduce_sum(
        # coarse_gating_distribution[:, :coarse_num_mixtures] * coarse_expert_distribution, 1)
        coarse_survived_experts[:, :coarse_num_mixtures] * coarse_expert_distribution, 1)
    coarse_probabilities = tf.reshape(coarse_probabilities_by_class_and_batch,
                                     [-1, 25])

    concat = tf.concat([model_input, coarse_probabilities], -1, name = 'middle_concat')
    # concat = tf.concat([reduced, coarse_probabilities], -1, name = 'middle_concat')

    ### Label Level
    label_gate_activations = slim.fully_connected(
        concat,
        vocab_size * (label_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_gates")
    label_expert_activations = slim.fully_connected(
        concat,
        vocab_size * label_num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_experts")

    label_gating_distribution = tf.nn.softmax(tf.reshape(
        label_gate_activations,
        [-1, label_num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    label_expert_distribution = tf.nn.sigmoid(tf.reshape(
        label_expert_activations,
        [-1, label_num_mixtures]))  # (Batch * #Labels) x num_mixtures

    ### Randomly sick
    survived_experts = slim.dropout(
        label_gating_distribution,
        keep_prob=0.75,
        noise_shape=tf.constant([1, label_num_mixtures + 1]),
        is_training=phase,
        scope="label_drop")

    label_probabilities_by_class_and_batch = tf.reduce_sum(
        # label_gating_distribution[:, :label_num_mixtures] * label_expert_distribution, 1)
        survived_experts[:, :label_num_mixtures] * label_expert_distribution, 1)
    label_probabilities = tf.reshape(label_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    matrix = np.load(FLAGS.refine)
    wij = tf.constant(matrix[:-2, :].T.astype('float32'))
    mn = tf.constant(matrix[-2, :].reshape(1, vocab_size).astype('float32') * 0.5)
    rg = tf.constant(1 / matrix[-1, :].reshape(1, vocab_size).astype('float32'))

    ### Refine the probability
    p1 = tf.matmul(label_probabilities, wij, name = "correlation_refine")
    # p1_centered = tf.subtract(p1, mn, name = "center_norm")
    # scale = tf.Variable(2.0, name = 'scale')
    # p1_normed = tf.multiply(p1, rg, name = "range_norm")
    # p1_rescaled = tf.sigmoid(tf.multiply(p1_normed, scale, name = "rescale"))
    p1_normed = slim.batch_norm(
        tf.reshape(p1, [-1, vocab_size]),
        center=True,
        scale=True,
        activation_fn=nn.sigmoid,
        is_training=phase,
        scope="bn")

    # return {"predictions": label_probabilities, "coarse_predictions": coarse_probabilities}
    return {"predictions": p1_normed, "coarse_predictions": coarse_probabilities}
