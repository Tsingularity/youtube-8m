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
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 8,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

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

class CCModel(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    reduce_size = 256
    state_size = 96 
    batch_size = tf.shape(model_input)[0]
    time_steps = vocab_size
    
    lstm = tf.contrib.rnn.LSTMCell(state_size)
    prob_chains = []

    reduced_feature = slim.fully_connected(
        model_input,
        reduce_size,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="reduce")
    state = lstm.zero_state(batch_size, dtype=tf.float32) 
    logistic_weights = tf.get_variable("logistic_weights", [state_size, 1], initializer=tf.random_normal_initializer())
    logistic_biases = tf.get_variable("logistic_biases", [1], initializer=tf.constant_initializer(0.0))

    for step in xrange(time_steps):
      if step > 0:
        tf.get_variable_scope().reuse_variables()
      output, state = lstm(reduced_feature, state) 
      binary_response = tf.sigmoid(tf.matmul(output, logistic_weights) + logistic_biases)
      prob_chains.append(binary_response)

    # paddings = tf.zeros([batch_size, vocab_size - time_steps], dtype=tf.float32)
    # prob_chains.append(paddings)

    final_probabilities = tf.concat(prob_chains, 1)
    return {"predictions": final_probabilities}

class PartlyCCModel(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    reduce_size = 512 
    state_size = 512 
    batch_size = tf.shape(model_input)[0]
    part_size = 36
    time_steps = vocab_size // part_size
    
    lstm = tf.contrib.rnn.LSTMCell(state_size)
    prob_chains = []

    reduced_feature = slim.fully_connected(
        model_input,
        reduce_size,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="reduce")
    state = lstm.zero_state(batch_size, dtype=tf.float32)
    logistic_weights = tf.get_variable("logistic_weights", [state_size, part_size], initializer=tf.random_normal_initializer())
    logistic_biases = tf.get_variable("logistic_biases", [part_size], initializer=tf.constant_initializer(0.0))

    for step in xrange(time_steps):
      if step > 0:
        tf.get_variable_scope().reuse_variables()
      output, state = lstm(reduced_feature, state) 
      binary_part_response = tf.sigmoid(tf.matmul(output, logistic_weights) + logistic_biases)
      prob_chains.append(binary_part_response)

    # paddings = tf.zeros([batch_size, vocab_size - time_steps], dtype=tf.float32)
    # prob_chains.append(paddings)

    final_probabilities = tf.concat(prob_chains, 1)
    return {"predictions": final_probabilities}

class PRCCModel(models.BaseModel): 
  """ 
  Partly Recurring Classifiers Chain
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    reduce_size = 512 
    state_size = 1024
    proj_size = 512
    batch_size = tf.shape(model_input)[0]
    part_size = 393 
    time_steps = vocab_size // part_size
    
    lstm = tf.contrib.rnn.LSTMCell(state_size, num_proj=proj_size)
    prob_chains = []

    reduced_feature = slim.fully_connected(
        model_input,
        reduce_size,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="reduce")
    state = lstm.zero_state(batch_size, dtype=tf.float32)
    part_response = tf.zeros([batch_size, part_size], dtype=tf.float32)
    # logistic_weights = tf.get_variable("logistic_weights", [state_size, part_size], initializer=tf.random_normal_initializer())
    # Newly
    logistic_weights = tf.get_variable("logistic_weights", [proj_size, part_size], initializer=tf.random_normal_initializer())
    #
    logistic_biases = tf.get_variable("logistic_biases", [part_size], initializer=tf.constant_initializer(0.0))

    for step in xrange(time_steps):
      if step > 0:
        tf.get_variable_scope().reuse_variables()
      output, state = lstm(tf.concat([reduced_feature, part_response], 1), state) 
      part_response = tf.sigmoid(tf.matmul(output, logistic_weights) + logistic_biases)
      prob_chains.append(part_response)

    final_probabilities = tf.concat(prob_chains, 1)
    return {"predictions": final_probabilities}

class PRCCConcatModel(models.BaseModel): 
  """ 
  Partly Recurring Classifiers Chain
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    reduced_feature_size = 1024 
    reduced_distr_size = 1024 
    state_size = 1024 
    batch_size = tf.shape(model_input)[0]
    part_size = 393 * 4 
    time_steps = vocab_size // part_size
    
    lstm = tf.contrib.rnn.LSTMCell(state_size)
    prob_chains = []
  
    # reduced input feature
    reduced_feature_activations = slim.fully_connected(
        model_input,
        reduced_feature_size,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="reduce")
    reduced_feature = tf.nn.relu(slim.batch_norm(reduced_feature_activations, is_training=is_training))
    # distribution feature
    distribution = tf.zeros([batch_size, vocab_size], dtype=tf.float32)

    state = lstm.zero_state(batch_size, dtype=tf.float32)
    logistic_weights = [tf.get_variable("logistic_weights_" + str(i), [state_size, part_size], initializer=tf.random_normal_initializer()) for i in range(time_steps)]
    logistic_biases = [tf.get_variable("logistic_biases_" + str(i), [part_size], initializer=tf.constant_initializer(0.0)) for i in range(time_steps)]
    distr_weights = tf.get_variable("distr_weights", [vocab_size, reduced_distr_size], initializer=tf.random_normal_initializer())
    distr_biases = tf.get_variable("distr_biases", [reduced_distr_size], initializer=tf.constant_initializer(0.0))

    for step in xrange(time_steps):
      if step > 0:
        tf.get_variable_scope().reuse_variables()
      # distribution reduced feature
      reduced_distr_activations = tf.matmul(distribution, distr_weights) + distr_biases 
      reduced_distr = tf.nn.relu(slim.batch_norm(reduced_distr_activations, is_training=is_training))
      #

      output, state = lstm(tf.concat([reduced_feature, reduced_distr], 1), state) 
      part_response = tf.sigmoid(tf.matmul(output, logistic_weights[step]) + logistic_biases[step])
      prob_chains.append(part_response)
      pre = distribution[:, :step * part_size] if step > 0 else None
      aft = distribution[:, (step + 1) * part_size:] if step + 1 < time_steps else None
      distribution = part_response
      if pre is not None:
        distribution = tf.concat([pre, distribution], 1)
      if aft is not None:
        distribution = tf.concat([distribution, aft], 1)

    final_probabilities = tf.concat(prob_chains, 1)
    return {"predictions": final_probabilities}

class CCMoeModel(models.BaseModel): 
  """ 
  Classifiers Chain Moe
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    reduced_feature_size = 512 
    reduced_distr_size = 1024 
    batch_size = tf.shape(model_input)[0]
    part_size = 393 
    time_steps = vocab_size // part_size
    num_experts = 8
    
    prob_chains = []
  
    # reduced input feature
    reduced_feature_activations = slim.fully_connected(
        model_input,
        reduced_feature_size,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="reduce")
    reduced_feature = tf.nn.relu(slim.batch_norm(reduced_feature_activations, is_training=is_training))
    # distribution feature
    distribution = tf.zeros([batch_size, vocab_size], dtype=tf.float32)

    distr_weights = tf.get_variable("distr_weights", [vocab_size, reduced_distr_size], initializer=tf.random_normal_initializer())
    distr_biases = tf.get_variable("distr_biases", [reduced_distr_size], initializer=tf.constant_initializer(0.0))

    for step in xrange(time_steps):
      if step > 0:
        tf.get_variable_scope().reuse_variables()
      # distribution reduced feature
      reduced_distr_activations = tf.matmul(distribution, distr_weights) + distr_biases 
      reduced_distr = tf.nn.relu(slim.batch_norm(reduced_distr_activations, is_training=is_training))
      #

      # group predictions
      # group_input = tf.concat([reduced_distr, reduced_feature], 1)
      group_input = slim.dropout(tf.concat([reduced_distr, reduced_feature], 1),
        keep_prob=0.5,
        is_training=is_training)
      group_expert_activations = slim.fully_connected(
        group_input,
        part_size * num_experts,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="pred_" + str(step)
      )
      group_gate_activations = slim.fully_connected(
        group_input,
        part_size * (num_experts + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gate_" + str(step) 
      )

      expert_distribution = tf.nn.sigmoid(tf.reshape(
        group_expert_activations,
        [-1, num_experts]))
      gate_distribution = tf.nn.softmax(tf.reshape(
        group_gate_activations,
        [-1, num_experts + 1]))

      expert_distr_by_class_and_batch = tf.reduce_sum(
          gate_distribution[:, :num_experts] * expert_distribution, 1)
      group_predictions = tf.reshape(expert_distr_by_class_and_batch,
          [-1, part_size])
      #

      prob_chains.append(group_predictions)
      pre = distribution[:, :step * part_size] if step > 0 else None
      aft = distribution[:, (step + 1) * part_size:] if step + 1 < time_steps else None
      distribution = group_predictions
      if pre is not None:
        distribution = tf.concat([pre, distribution], 1)
      if aft is not None:
        distribution = tf.concat([distribution, aft], 1)

    final_probabilities = tf.concat(prob_chains, 1)
    return {"predictions": final_probabilities}

class CCOrigMoeModel(models.BaseModel): 
  """ 
  Classifiers Chain Moe
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    # reduced_feature_size = 512 
    # reduced_distr_size = 1024 
    batch_size = tf.shape(model_input)[0]
    part_size = 393 
    time_steps = vocab_size // part_size
    num_experts = 8
    
    prob_chains = []
  
    # reduced input feature
    # reduced_feature_activations = slim.fully_connected(
    #     model_input,
    #     reduced_feature_size,
    #     activation_fn=None,
    #     weights_regularizer=slim.l2_regularizer(l2_penalty),
    #     scope="reduce")
    # reduced_feature = tf.nn.relu(slim.batch_norm(reduced_feature_activations, is_training=is_training))
    # distribution feature
    distribution = tf.zeros([batch_size, vocab_size], dtype=tf.float32)

    # distr_weights = tf.get_variable("distr_weights", [vocab_size, reduced_distr_size], initializer=tf.random_normal_initializer())
    # distr_biases = tf.get_variable("distr_biases", [reduced_distr_size], initializer=tf.constant_initializer(0.0))

    for step in xrange(time_steps):
      if step > 0:
        tf.get_variable_scope().reuse_variables()
      # distribution reduced feature
      # reduced_distr_activations = tf.matmul(distribution, distr_weights) + distr_biases 
      # reduced_distr = tf.nn.relu(slim.batch_norm(reduced_distr_activations, is_training=is_training))
      #

      # group predictions
      # group_input = tf.concat([reduced_distr, reduced_feature], 1)
      group_input = tf.concat([distribution, model_input], 1)
      # group_input = slim.dropout(tf.concat([reduced_distr, reduced_feature], 1),
      #   keep_prob=0.5,
      #   is_training=is_training)
      group_expert_activations = slim.fully_connected(
        group_input,
        part_size * num_experts,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="pred_" + str(step)
      )
      group_gate_activations = slim.fully_connected(
        group_input,
        part_size * (num_experts + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gate_" + str(step) 
      )

      expert_distribution = tf.nn.sigmoid(tf.reshape(
        group_expert_activations,
        [-1, num_experts]))
      gate_distribution = tf.nn.softmax(tf.reshape(
        group_gate_activations,
        [-1, num_experts + 1]))

      expert_distr_by_class_and_batch = tf.reduce_sum(
          gate_distribution[:, :num_experts] * expert_distribution, 1)
      group_predictions = tf.reshape(expert_distr_by_class_and_batch,
          [-1, part_size])
      #

      prob_chains.append(group_predictions)
      pre = distribution[:, :step * part_size] if step > 0 else None
      aft = distribution[:, (step + 1) * part_size:] if step + 1 < time_steps else None
      distribution = group_predictions
      if pre is not None:
        distribution = tf.concat([pre, distribution], 1)
      if aft is not None:
        distribution = tf.concat([distribution, aft], 1)

    final_probabilities = tf.concat(prob_chains, 1)
    return {"predictions": final_probabilities}

class CCDistrMoeModel(models.BaseModel): 
  """ 
  Classifiers Chain Moe
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    # reduced_feature_size = 512 
    reduced_distr_size = 1024 
    batch_size = tf.shape(model_input)[0]
    part_size = 393 
    time_steps = vocab_size // part_size
    num_experts = 8
    
    prob_chains = []
  
    # reduced input feature
    # reduced_feature_activations = slim.fully_connected(
    #     model_input,
    #     reduced_feature_size,
    #     activation_fn=None,
    #     weights_regularizer=slim.l2_regularizer(l2_penalty),
    #     scope="reduce")
    # reduced_feature = tf.nn.relu(slim.batch_norm(reduced_feature_activations, is_training=is_training))
    # distribution feature
    distribution = tf.zeros([batch_size, vocab_size], dtype=tf.float32)

    distr_weights = tf.get_variable("distr_weights", [vocab_size, reduced_distr_size], initializer=tf.random_normal_initializer())
    distr_biases = tf.get_variable("distr_biases", [reduced_distr_size], initializer=tf.constant_initializer(0.0))

    for step in xrange(time_steps):
      if step > 0:
        tf.get_variable_scope().reuse_variables()
      # distribution reduced feature
      reduced_distr_activations = tf.matmul(distribution, distr_weights) + distr_biases 
      reduced_distr = tf.nn.relu(slim.batch_norm(reduced_distr_activations, is_training=is_training))
      #

      # group predictions
      # group_input = tf.concat([reduced_distr, reduced_feature], 1)
      # group_input = tf.concat([distribution, model_input], 1)
      group_input = tf.concat([reduced_distr, model_input], 1)
      # group_input = slim.dropout(tf.concat([reduced_distr, reduced_feature], 1),
      #   keep_prob=0.5,
      #   is_training=is_training)
      group_expert_activations = slim.fully_connected(
        group_input,
        part_size * num_experts,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="pred_" + str(step)
      )
      group_gate_activations = slim.fully_connected(
        group_input,
        part_size * (num_experts + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gate_" + str(step) 
      )

      expert_distribution = tf.nn.sigmoid(tf.reshape(
        group_expert_activations,
        [-1, num_experts]))
      gate_distribution = tf.nn.softmax(tf.reshape(
        group_gate_activations,
        [-1, num_experts + 1]))

      expert_distr_by_class_and_batch = tf.reduce_sum(
          gate_distribution[:, :num_experts] * expert_distribution, 1)
      group_predictions = tf.reshape(expert_distr_by_class_and_batch,
          [-1, part_size])
      #

      prob_chains.append(group_predictions)
      pre = distribution[:, :step * part_size] if step > 0 else None
      aft = distribution[:, (step + 1) * part_size:] if step + 1 < time_steps else None
      distribution = group_predictions
      if pre is not None:
        distribution = tf.concat([pre, distribution], 1)
      if aft is not None:
        distribution = tf.concat([distribution, aft], 1)

    final_probabilities = tf.concat(prob_chains, 1)
    return {"predictions": final_probabilities}
