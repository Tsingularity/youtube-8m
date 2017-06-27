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

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("window",3,"window size for cnn")
flags.DEFINE_integer("ac",128,"channel num for audio cnn")
flags.DEFINE_integer("vc",1024,"channel num for visual cnn")
flags.DEFINE_integer("poolsize",3,"size of max pool")
flags.DEFINE_integer("visual_lstm_cells",1024,"Number of cells for visual info")
flags.DEFINE_integer("audio_lstm_cells",128,"Number of cells for audio info")
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 1, "Number of LSTM layers.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BiLstmModel(models.BaseModel):
	def create_model(self,model_input,vocab_size,num_frames,**unused_params):
		
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers

		stacked_lstm_forward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(
					lstm_size,forget_bias=1.0,state_is_tuple=False)
				for _ in range(number_of_layers)
				], state_is_tuple=False)
		
		stacked_lstm_backward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(
					lstm_size,forget_bias=1.0,state_is_tuple=False)
				for _ in range(number_of_layers)
				], state_is_tuple=False)

		outputs,state = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_forward,stacked_lstm_backward,model_input,sequence_length=num_frames,dtype=tf.float32)
		
		#concat_state = tf.concat(state,1)
		concat_state = tf.add(state[0],state[1])

		aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(model_input=concat_state,vocab_size=vocab_size,**unused_params)


class ATT_BiLstm(models.BaseModel):
	def create_model(self,model_input,vocab_size,num_frames,**unused_params):
		
		shape = tf.shape(model_input)
		#print 'MODEL INPUT SHAPE !!!!'
		#print shape
		batch_size = tf.shape(model_input)[0]
		max_frames = tf.shape(model_input)[1]
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers

		stacked_gru_forward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
			])
		
		stacked_gru_backward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
			])
		
		outputs,state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_forward,stacked_gru_backward,model_input,sequence_length=num_frames,dtype=tf.float32)
		
		#print 'OUTPUTS SHAPE !!!'
		#print tf.shape(outputs)

		output_forward = outputs[0]
		output_backward = outputs[1]
		
		#print 'output_forward shape !!!'
		#print tf.shape(output_forward)
		

		output = tf.add(output_forward,output_backward)

		attention_w = tf.get_variable('attention_w',[lstm_size,1])
		

		temp_att = tf.reshape(tf.tanh(output),[batch_size*max_frames,lstm_size])

		temp_att = tf.reshape(tf.matmul(temp_att,attention_w),[batch_size,max_frames])

		temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames])

		temp_att = tf.reshape(tf.matmul(temp_att,output),[batch_size,lstm_size])
		
		temp_att = tf.tanh(temp_att)

		aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(model_input=temp_att,vocab_size=vocab_size,**unused_params)


class DropOut_ATT_BiLSTM(models.BaseModel):
	def create_model(self,model_input,vocab_size,num_frames,**unused_params):
		
		shape = tf.shape(model_input)
		#print 'MODEL INPUT SHAPE !!!!'
		#print shape
		batch_size = tf.shape(model_input)[0]
		max_frames = tf.shape(model_input)[1]
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers
		
		stacked_lstm_forward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(lstm_size) for _ in range(number_of_layers)
			])
		
		stacked_lstm_backward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(lstm_size) for _ in range(number_of_layers)
			])
		
		if FLAGS.is_training:
				stacked_lstm_forward = tf.contrib.rnn.DropoutWrapper(stacked_lstm_forward)
				stacked_lstm_backward = tf.contrib.rnn.DropoutWrapper(stacked_lstm_backward)


		outputs,state = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_forward,stacked_lstm_backward,model_input,sequence_length=num_frames,dtype=tf.float32)
		
		#print 'OUTPUTS SHAPE !!!'
		#print tf.shape(outputs)

		output_forward = outputs[0]
		output_backward = outputs[1]
		
		#print 'output_forward shape !!!'
		#print tf.shape(output_forward)
		

		output = tf.add(output_forward,output_backward)

		attention_w = tf.get_variable('attention_w',[lstm_size,1])
		

		temp_att = tf.reshape(tf.tanh(output),[batch_size*max_frames,lstm_size])

		temp_att = tf.reshape(tf.matmul(temp_att,attention_w),[batch_size,max_frames])

		temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames])

		temp_att = tf.reshape(tf.matmul(temp_att,output),[batch_size,lstm_size])
		
		temp_att = tf.tanh(temp_att)

		aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(model_input=temp_att,vocab_size=vocab_size,**unused_params)


class Split_ATT_BiLstm(models.BaseModel):
	def create_model(self,model_input,vocab_size,num_frames,**unused_params):
		
		shape = tf.shape(model_input)
		#print 'MODEL INPUT SHAPE !!!!'
		#print shape
		batch_size = tf.shape(model_input)[0]
		max_frames = tf.shape(model_input)[1]
		visual_lstm_size = FLAGS.visual_lstm_cells
		audio_lstm_size = FLAGS.audio_lstm_cells

		visual_input = model_input[:,:,:1024]
		audio_input = model_input[:,:,1024:1024+128]


		number_of_layers = FLAGS.lstm_layers

		visual_stacked_lstm_forward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(visual_lstm_size) for _ in range(number_of_layers)
			])
		
		visual_stacked_lstm_backward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(visual_lstm_size) for _ in range(number_of_layers)
			])
		
		with tf.variable_scope('visual_lstm'):		
			visual_outputs,visual_state = tf.nn.bidirectional_dynamic_rnn(visual_stacked_lstm_forward,visual_stacked_lstm_backward,visual_input,sequence_length=num_frames,dtype=tf.float32)
		
		audio_stacked_lstm_forward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(audio_lstm_size) for _ in range(number_of_layers)
			])
		
		audio_stacked_lstm_backward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(audio_lstm_size) for _ in range(number_of_layers)
			])
		
		with tf.variable_scope('audio_lstm'):	
			audio_outputs,audio_state = tf.nn.bidirectional_dynamic_rnn(audio_stacked_lstm_forward,audio_stacked_lstm_backward,audio_input,sequence_length=num_frames,dtype=tf.float32)
	
		#print 'OUTPUTS SHAPE !!!'
		#print tf.shape(outputs)

		visual_output_forward = visual_outputs[0]
		visual_output_backward = visual_outputs[1]

		audio_output_forward = audio_outputs[0]
		audio_output_backward = audio_outputs[1]
		
		#print 'output_forward shape !!!'
		#print tf.shape(output_forward)
		

		visual_output = tf.add(visual_output_forward,visual_output_backward)
		audio_output = tf.add(audio_output_forward,audio_output_backward)

		attention_visual = tf.get_variable('attention_visual',[visual_lstm_size,1])
		
		temp_att = tf.reshape(tf.tanh(visual_output),[batch_size*max_frames,visual_lstm_size])

		temp_att = tf.reshape(tf.matmul(temp_att,attention_visual),[batch_size,max_frames])

		temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames])

		temp_att = tf.reshape(tf.matmul(temp_att,visual_output),[batch_size,visual_lstm_size])
		
		visual_att = tf.tanh(temp_att)
		
		attention_audio = tf.get_variable('attention_audio',[audio_lstm_size,1])
		
		temp_att = tf.reshape(tf.tanh(audio_output),[batch_size*max_frames,audio_lstm_size])

		temp_att = tf.reshape(tf.matmul(temp_att,attention_audio),[batch_size,max_frames])

		temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames])

		temp_att = tf.reshape(tf.matmul(temp_att,audio_output),[batch_size,audio_lstm_size])
		
		audio_att = tf.tanh(temp_att)

		final_att = tf.concat([visual_att,audio_att],1)

		aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(model_input=final_att,vocab_size=vocab_size,**unused_params)


class Maxpool_ATT_BiLstm(models.BaseModel):
	def create_model(self,model_input,vocab_size,num_frames,**unused_params):
		
		shape = tf.shape(model_input)
		#print 'MODEL INPUT SHAPE !!!!'
		#print shape
		batch_size = tf.shape(model_input)[0]
		max_frames = tf.shape(model_input)[1]
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers
		with tf.variable_scope('local_lstm'):
			#with tf.variable_scope('forward'):
			stacked_gru_forward = tf.contrib.rnn.MultiRNNCell(
				[
					tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
				])
		
			stacked_gru_backward = tf.contrib.rnn.MultiRNNCell(
				[
					tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
				])
		
			outputs,state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_forward,stacked_gru_backward,model_input,sequence_length=num_frames,dtype=tf.float32)
		
		#print 'OUTPUTS SHAPE !!!'
		#print tf.shape(outputs)

		output_forward = outputs[0]
		output_backward = outputs[1]
		
		#print 'output_forward shape !!!'
		#print tf.shape(output_forward)
		

		output = tf.add(output_forward,output_backward)
		
		#maxpooling#######
		ksize = FLAGS.poolsize
		output = tf.expand_dims(output,-1)
		output = tf.nn.max_pool(output,ksize=[1,ksize,1,1],strides=[1,ksize,1,1],padding='SAME')
		output = tf.reshape(output,[batch_size,max_frames/ksize,lstm_size])

		with tf.variable_scope('postmax_lstm'):
			lstm_forward = tf.contrib.rnn.MultiRNNCell(
				[
					tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
				])
		
			lstm_backward = tf.contrib.rnn.MultiRNNCell(
				[
					tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
				])

			new_len = max_frames/ksize*tf.ones_like(num_frames)
			#new_len = tf.multiply(max_frames/ksize,tf.ones(batch_size))
			
			outputs,state = tf.nn.bidirectional_dynamic_rnn(lstm_forward,lstm_backward,output,sequence_length=new_len,dtype=tf.float32)

		post_output_forward = outputs[0]
		post_output_backward = outputs[1]
		
		output = tf.add(post_output_forward,post_output_backward)

		attention_w = tf.get_variable('attention_w',[lstm_size,1])
		

		temp_att = tf.reshape(tf.tanh(output),[batch_size*max_frames/ksize,lstm_size])

		temp_att = tf.reshape(tf.matmul(temp_att,attention_w),[batch_size,max_frames/ksize])

		temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames/ksize])

		temp_att = tf.reshape(tf.matmul(temp_att,output),[batch_size,lstm_size])
		
		temp_att = tf.tanh(temp_att)

		aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(model_input=temp_att,vocab_size=vocab_size,**unused_params)

	
class CNN(models.BaseModel):
	def create_model(self,model_input,vocab_size,num_frames,keep_prob,**unused_params):
		
		batch_size = tf.shape(model_input)[0]
		max_frames = tf.shape(model_input)[1]
		
		visual_input = model_input[:,:,:1024]
		audio_input = model_input[:,:,1024:1024+128]

		visual_input = tf.expand_dims(visual_input,-1)
		audio_input = tf.expand_dims(audio_input,-1)

		visual_cnn = slim.conv2d(inputs=visual_input,num_outputs=FLAGS.vc,kernel_size=[FLAGS.window,1024],stride=[1,1024],scope='visual_cnn',padding='SAME',biases_initializer=None,activation_fn=None)
		audio_cnn = slim.conv2d(inputs=audio_input,num_outputs=FLAGS.ac,kernel_size=[FLAGS.window,128],stride=[1,128],scope='audio_cnn',padding='SAME',biases_initializer=None,activation_fn=None)
		
		visual_out = tf.reshape(visual_cnn,[batch_size,max_frames,FLAGS.vc,1])
		audio_out = tf.reshape(audio_cnn,[batch_size,max_frames,FLAGS.ac,1])

		visual_pool = tf.nn.max_pool(visual_out,ksize=[1,300,1,1],strides=[1,300,1,1],padding='SAME')
		audio_pool = tf.nn.max_pool(audio_out,ksize=[1,300,1,1],strides=[1,300,1,1],padding='SAME')

		visual_output = tf.reshape(visual_pool,[batch_size,FLAGS.vc])
		audio_output = tf.reshape(audio_pool,[batch_size,FLAGS.ac])

		output = tf.concat([visual_output,audio_output],1)
		output = tf.nn.relu(output)

		aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(model_input=output,vocab_size=vocab_size,keep_prob=keep_prob,**unused_params)


class Split_BiLstm_CNN(models.BaseModel):
	def create_model(self,model_input,vocab_size,num_frames,**unused_params):
		
		shape = tf.shape(model_input)
		batch_size = tf.shape(model_input)[0]
		max_frames = tf.shape(model_input)[1]
		visual_lstm_size = FLAGS.visual_lstm_cells
		audio_lstm_size = FLAGS.audio_lstm_cells

		visual_input = model_input[:,:,:1024]
		audio_input = model_input[:,:,1024:1024+128]


		number_of_layers = FLAGS.lstm_layers

		visual_stacked_lstm_forward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(visual_lstm_size) for _ in range(number_of_layers)
			])
		
		visual_stacked_lstm_backward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(visual_lstm_size) for _ in range(number_of_layers)
			])
		
		with tf.variable_scope('visual_lstm'):		
			visual_outputs,visual_state = tf.nn.bidirectional_dynamic_rnn(visual_stacked_lstm_forward,visual_stacked_lstm_backward,visual_input,sequence_length=num_frames,dtype=tf.float32)
		
		audio_stacked_lstm_forward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(audio_lstm_size) for _ in range(number_of_layers)
			])
		
		audio_stacked_lstm_backward = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.LSTMCell(audio_lstm_size) for _ in range(number_of_layers)
			])
		
		with tf.variable_scope('audio_lstm'):	
			audio_outputs,audio_state = tf.nn.bidirectional_dynamic_rnn(audio_stacked_lstm_forward,audio_stacked_lstm_backward,audio_input,sequence_length=num_frames,dtype=tf.float32)
	
		#print 'OUTPUTS SHAPE !!!'
		#print tf.shape(outputs)

		visual_output_forward = visual_outputs[0]
		visual_output_backward = visual_outputs[1]

		audio_output_forward = audio_outputs[0]
		audio_output_backward = audio_outputs[1]
		
		#print 'output_forward shape !!!'
		#print tf.shape(output_forward)
		

		visual_output = tf.add(visual_output_forward,visual_output_backward)
		audio_output = tf.add(audio_output_forward,audio_output_backward)

		visual_input = tf.expand_dims(visual_output,-1)
		audio_input = tf.expand_dims(audio_output,-1)

		visual_cnn = slim.conv2d(inputs=visual_input,num_outputs=FLAGS.vc,kernel_size=[FLAGS.window,1024],stride=[1,1024],scope='visual_cnn',padding='SAME',biases_initializer=None,activation_fn=None)
		audio_cnn = slim.conv2d(inputs=audio_input,num_outputs=FLAGS.ac,kernel_size=[FLAGS.window,128],stride=[1,128],scope='audio_cnn',padding='SAME',biases_initializer=None,activation_fn=None)
		
		visual_out = tf.reshape(visual_cnn,[batch_size,max_frames,FLAGS.vc,1])
		audio_out = tf.reshape(audio_cnn,[batch_size,max_frames,FLAGS.ac,1])

		visual_pool = tf.nn.max_pool(visual_out,ksize=[1,300,1,1],strides=[1,300,1,1],padding='SAME')
		audio_pool = tf.nn.max_pool(audio_out,ksize=[1,300,1,1],strides=[1,300,1,1],padding='SAME')

		visual_output = tf.reshape(visual_pool,[batch_size,FLAGS.vc])
		audio_output = tf.reshape(audio_pool,[batch_size,FLAGS.ac])

		output = tf.concat([visual_output,audio_output],1)
		output = tf.nn.relu(output)

		aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(model_input=output,vocab_size=vocab_size,**unused_params)

class new_Maxpool_ATT_BiLstm(models.BaseModel):
  def create_model(self,model_input,vocab_size,num_frames,keep_prob,**unused_params):

    #model_input = slim.dropout(model_input,keep_prob=keep_prob)
    shape = tf.shape(model_input)
    #print 'MODEL INPUT SHAPE !!!!'
    #print shape
    batch_size = tf.shape(model_input)[0]
    max_frames = tf.shape(model_input)[1]
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    with tf.variable_scope('local_lstm'):
      #with tf.variable_scope('forward'):
      stacked_gru_forward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      stacked_gru_backward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      outputs,state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_forward,stacked_gru_backward,model_input,sequence_length=num_frames,dtype=tf.float32)

    #print 'OUTPUTS SHAPE !!!'
    #print tf.shape(outputs)

    output_forward = outputs[0]
    output_backward = outputs[1]

    #print 'output_forward shape !!!'
    #print tf.shape(output_forward)


    output = tf.add(output_forward,output_backward)

    #maxpooling#######
    ksize = FLAGS.poolsize
    output = tf.expand_dims(output,-1)
    output = tf.nn.max_pool(output,ksize=[1,ksize,1,1],strides=[1,ksize,1,1],padding='SAME')
    output = tf.reshape(output,[batch_size,max_frames/ksize,lstm_size])

    with tf.variable_scope('postmax_lstm'):
      lstm_forward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      lstm_backward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      new_len = max_frames/ksize*tf.ones_like(num_frames)
      #new_len = tf.multiply(max_frames/ksize,tf.ones(batch_size))
      
      post_outputs,post_state = tf.nn.bidirectional_dynamic_rnn(lstm_forward,lstm_backward,output,sequence_length=new_len,dtype=tf.float32)

    post_output_forward = post_outputs[0]
    post_output_backward = post_outputs[1]

    output = tf.add(post_output_forward,post_output_backward)

    attention_w = tf.get_variable('attention_w',[lstm_size,1])


    temp_att = tf.reshape(tf.tanh(output),[batch_size*max_frames/ksize,lstm_size])

    temp_att = tf.reshape(tf.matmul(temp_att,attention_w),[batch_size,max_frames/ksize])

    temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames/ksize])

    temp_att = tf.reshape(tf.matmul(temp_att,output),[batch_size,lstm_size])

    temp_att = tf.tanh(temp_att)

    aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(model_input=temp_att,vocab_size=vocab_size,keep_prob=keep_prob,**unused_params)

class Random(models.BaseModel):

  def create_model(self,model_input,vocab_size,num_frames,keep_prob,**unused_params):
    
    batch_size = tf.shape(model_input)[0]
    #batch_size = model_input.get_shape().as_list()[0]
    #batch_size = FLAGS.batch_size
    
    #print 'DA SHA BI '+str(batch_size)
    #print model_input.get_shape().as_list()
    #print num_frames.get_shape().as_list()
    starts = tf.range(0,300,5)
    randoms = tf.random_uniform([batch_size,60],0,5,dtype=tf.int32)
    frame_index = tf.add(starts,randoms)
    
    batch_index = tf.tile(tf.expand_dims(tf.range(batch_size),1),[1,60])

    index = tf.stack([batch_index,frame_index],2)
    new_input = tf.gather_nd(model_input,index)

    #new_input = [[None]*60]*tf.shape(model_input)[0]
    #new_input = tf.zeros([batch_size,60,1152])
    #new_input = []
    #for i in range(batch_size):
    #  one_input = []
    #  for j in range(60):
    #    one_input.append(model_input[i][indexs[i][j]])
    #  new_input.append(one_input)

    new_input = tf.convert_to_tensor(new_input)

    batch_size = tf.shape(new_input)[0]
    max_frames = tf.shape(new_input)[1]
    
    #new_frames = [max_frames]*batch_size
    new_frames = tf.cast(tf.cast(num_frames,tf.float32)/5,tf.int32)
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    
    with tf.variable_scope("forward"):
      stacked_lstm_forward = tf.contrib.rnn.MultiRNNCell(
        [
          #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(lstm_size),output_keep_prob=keep_prob) for _ in range(number_of_layers)
          tf.contrib.rnn.LSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

    with tf.variable_scope("backward"):
      stacked_lstm_backward = tf.contrib.rnn.MultiRNNCell(
        [
          #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(lstm_size),output_keep_prob=keep_prob) for _ in range(number_of_layers)
          tf.contrib.rnn.LSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

    with tf.variable_scope("bi"):
      outputs,state = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_forward,stacked_lstm_backward,new_input,sequence_length=new_frames,dtype=tf.float32)
    
    output_forward = outputs[0]
    output_backward = outputs[1]

    output = tf.add(output_forward,output_backward)

    attention_w = tf.get_variable('attention_w',[lstm_size,1])
    temp_att = tf.reshape(tf.tanh(output),[batch_size*max_frames,lstm_size])
    temp_att = tf.reshape(tf.matmul(temp_att,attention_w),[batch_size,max_frames])
    temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames])
    temp_att = tf.reshape(tf.matmul(temp_att,output),[batch_size,lstm_size])
    temp_att = tf.tanh(temp_att)

    aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(model_input=temp_att,vocab_size=vocab_size,keep_prob=keep_prob,**unused_params)


class zhao_Maxpool_ATT_BiLstm(models.BaseModel):
	def create_model(self,model_input,vocab_size,num_frames,**unused_params):
		
		shape = tf.shape(model_input)
		#print 'MODEL INPUT SHAPE !!!!'
		#print shape
		batch_size = tf.shape(model_input)[0]
		max_frames = tf.shape(model_input)[1]
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers
		with tf.variable_scope('local_lstm'):
			#with tf.variable_scope('forward'):
			stacked_gru_forward = tf.contrib.rnn.MultiRNNCell(
				[
					tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
				])
		
			stacked_gru_backward = tf.contrib.rnn.MultiRNNCell(
				[
					tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
				])
		
			outputs,state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_forward,stacked_gru_backward,model_input,sequence_length=num_frames,dtype=tf.float32)
		
		#print 'OUTPUTS SHAPE !!!'
		#print tf.shape(outputs)

		output_forward = outputs[0]
		output_backward = outputs[1]
		
		#print 'output_forward shape !!!'
		#print tf.shape(output_forward)
		

		output = tf.add(output_forward,output_backward)
		
		#maxpooling#######
		ksize = FLAGS.poolsize
		output = tf.expand_dims(output,-1)
		output = tf.nn.max_pool(output,ksize=[1,ksize,1,1],strides=[1,ksize,1,1],padding='SAME')
		output = tf.reshape(output,[batch_size,max_frames/ksize,lstm_size])

		with tf.variable_scope('postmax_lstm'):
			lstm_forward = tf.contrib.rnn.MultiRNNCell(
				[
					tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
				])
		
			lstm_backward = tf.contrib.rnn.MultiRNNCell(
				[
					tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
				])

			num_len = max_frames/ksize*tf.ones_like(num_frames)
			#new_len = tf.multiply(max_frames/ksize,tf.ones(batch_size))
			
			pool_outputs,pool_state = tf.nn.bidirectional_dynamic_rnn(lstm_forward,lstm_backward,output,sequence_length=num_len,dtype=tf.float32)

		post_output_forward = pool_outputs[0]
		post_output_backward = pool_outputs[1]
		
		output = tf.add(post_output_forward,post_output_backward)

		attention_w = tf.get_variable('attention_w',[lstm_size,1])
		

		temp_att = tf.reshape(tf.tanh(output),[batch_size*max_frames/ksize,lstm_size])

		temp_att = tf.reshape(tf.matmul(temp_att,attention_w),[batch_size,max_frames/ksize])

		temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames/ksize])

		temp_att = tf.reshape(tf.matmul(temp_att,output),[batch_size,lstm_size])
		
		temp_att = tf.tanh(temp_att)

		aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(model_input=temp_att,vocab_size=vocab_size,**unused_params)

class hlstm(models.BaseModel):
  def create_model(self,model_input,vocab_size,num_frames,keep_prob,**unused_params):

    #model_input = slim.dropout(model_input,keep_prob=keep_prob)
    #shape = tf.shape(model_input)
    #print 'MODEL INPUT SHAPE !!!!'
    #print shape
    batch_size = tf.shape(model_input)[0]
    #max_frames = tf.shape(model_input)[1]
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    
#    first_input = tf.split(model_input,60,1)

########first_local_lstm#############
    first_input = tf.reshape(model_input,[batch_size*60,5,1152])
    first_len = 5*tf.ones([batch_size*60],tf.int32)

    with tf.variable_scope('local_lstm_1'):
      #with tf.variable_scope('forward'):
      stacked_gru_forward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      stacked_gru_backward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      first_outputs,first_state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_forward,stacked_gru_backward,first_input,sequence_length=first_len,dtype=tf.float32)

    #print 'OUTPUTS SHAPE !!!'
    #print tf.shape(outputs)

    #output_forward = state[0]
    #output_backward = state[1]

    #print 'output_forward shape !!!'
    #print tf.shape(output_forward)


    first_output = tf.add(first_state[0][-1].c,first_state[1][-1].c)

########second_local_lstm########
    second_input = tf.reshape(first_output,[batch_size*12,5,1152])
    second_len = 5*tf.ones([batch_size*12],tf.int32)

    with tf.variable_scope('local_lstm_2'):
      #with tf.variable_scope('forward'):
      stacked_gru_forward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      stacked_gru_backward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      second_outputs,second_state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_forward,stacked_gru_backward,second_input,sequence_length=second_len,dtype=tf.float32)

    second_output = tf.add(second_state[0][-1].c,second_state[1][-1].c)


########second_local_lstm_end####



    #ksize = FLAGS.poolsize
    #output = tf.expand_dims(output,-1)
    #output = tf.nn.max_pool(output,ksize=[1,ksize,1,1],strides=[1,ksize,1,1],padding='SAME')
    #output = tf.reshape(output,[batch_size,max_frames/ksize,lstm_size])
    global_input = tf.reshape(second_output,[batch_size,12,1152])
    global_len = 12*tf.ones([batch_size],tf.int32)

    with tf.variable_scope('global_lstm'):
      lstm_forward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      lstm_backward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      #new_len = max_frames/ksize*tf.ones_like(num_frames)
      #new_len = tf.multiply(max_frames/ksize,tf.ones(batch_size))
      
      global_outputs,global_state = tf.nn.bidirectional_dynamic_rnn(lstm_forward,lstm_backward,global_input,sequence_length=global_len,dtype=tf.float32)

    #post_output_forward = post_outputs[0]
    #post_output_backward = post_outputs[1]

    global_output = tf.add(global_outputs[0],global_outputs[1])

    attention_w = tf.get_variable('attention_w',[lstm_size,1])


    temp_att = tf.reshape(tf.tanh(global_output),[batch_size*12,lstm_size])

    temp_att = tf.reshape(tf.matmul(temp_att,attention_w),[batch_size,12])

    temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,12])

    temp_att = tf.reshape(tf.matmul(temp_att,global_output),[batch_size,lstm_size])

    temp_att = tf.tanh(temp_att)

    aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(model_input=temp_att,vocab_size=vocab_size,keep_prob=keep_prob,**unused_params)

class multi_att(models.BaseModel):
  def create_model(self,model_input,vocab_size,num_frames,keep_prob,**unused_params):

    #model_input = slim.dropout(model_input,keep_prob=keep_prob)
    shape = tf.shape(model_input)
    #print 'MODEL INPUT SHAPE !!!!'
    #print shape
    batch_size = tf.shape(model_input)[0]
    max_frames = tf.shape(model_input)[1]
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    with tf.variable_scope('local_lstm'):
      #with tf.variable_scope('forward'):
      stacked_gru_forward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      stacked_gru_backward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      outputs,state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_forward,stacked_gru_backward,model_input,sequence_length=num_frames,dtype=tf.float32)

    #print 'OUTPUTS SHAPE !!!'
    #print tf.shape(outputs)

    output_forward = outputs[0]
    output_backward = outputs[1]

    #print 'output_forward shape !!!'
    #print tf.shape(output_forward)


    output = tf.add(output_forward,output_backward)

    #maxpooling#######
    ksize = FLAGS.poolsize
    output = tf.expand_dims(output,-1)
    output = tf.nn.max_pool(output,ksize=[1,ksize,1,1],strides=[1,ksize,1,1],padding='SAME')
    output = tf.reshape(output,[batch_size,max_frames/ksize,lstm_size])

    with tf.variable_scope('postmax_lstm'):
      lstm_forward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      lstm_backward = tf.contrib.rnn.MultiRNNCell(
        [
          tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
        ])

      new_len = max_frames/ksize*tf.ones_like(num_frames)
      #new_len = tf.multiply(max_frames/ksize,tf.ones(batch_size))
      
      post_outputs,post_state = tf.nn.bidirectional_dynamic_rnn(lstm_forward,lstm_backward,output,sequence_length=new_len,dtype=tf.float32)

    post_output_forward = post_outputs[0]
    post_output_backward = post_outputs[1]

    output = tf.add(post_output_forward,post_output_backward)

    attention_w = tf.get_variable('attention_w',[lstm_size,25])


    temp_att = tf.reshape(tf.tanh(output),[batch_size*max_frames/ksize,lstm_size])

    temp_att = tf.reshape(tf.matmul(temp_att,attention_w),[batch_size,max_frames/ksize,25])

    temp_att = tf.transpose(tf.nn.softmax(temp_att,1),[0,2,1])

    laotie_25 = tf.reshape(tf.matmul(temp_att,output),[batch_size*25,lstm_size])

    belongs = tf.constant(np.load("/mnt/lustre/89share2/tangluming/youtube-8m/belongs.npy"))

    bg = tf.tile(tf.reshape(tf.range(start=0,limit=batch_size*25,delta=25),[batch_size,1]),[1,4716])

    ids = belongs+bg

    laotie = tf.reshape(tf.nn.embedding_lookup(laotie_25,ids),[batch_size*4716,1,lstm_size])
    
    laotie = slim.dropout(laotie,keep_prob=keep_prob)

    tang = tf.get_variable("tang",[4716,lstm_size,1])

    tang_copy = tf.tile(tang,[batch_size,1,1])

    final_out = tf.reshape(tf.sigmoid(tf.matmul(laotie,tang_copy)),[batch_size,4716])
    

#    temp_att = tf.reshape(tf.nn.softmax(temp_att),[batch_size,1,max_frames/ksize])

#    temp_att = tf.reshape(tf.matmul(temp_att,output),[batch_size,lstm_size])

#    temp_att = tf.tanh(temp_att)

    #aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model)
    #return aggregated_model().create_model(model_input=temp_att,vocab_size=vocab_size,keep_prob=keep_prob,**unused_params)
    return {"predictions":final_out,
            "features":laotie}


