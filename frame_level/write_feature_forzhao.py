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

"""Binary for generating predictions over a set of videos."""

import os
import time

import numpy
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import eval_util
import losses
import readers
import utils

FLAGS = flags.FLAGS

if __name__ == '__main__':
  
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to load the model files from.")
  flags.DEFINE_bool("is_training",False,"now in test phase")	

  flags.DEFINE_string("output_file", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("checkpoint_name","","inference checkpoint name")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_integer(
      "batch_size", 8192,
      "How many examples to process per batch.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")


  # Other flags.
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("top_k", 20,
                       "How many predictions to output per video.")
  flags.DEFINE_string("feature_name","tang","name of feature to write")

def format_lines(video_ids, features, labels):
  batch_size = len(video_ids)
  for video_index in range(batch_size):
#    top_indices = numpy.argpartition(predictions[video_index], -top_k)[-top_k:]
#    line = [(class_index, predictions[video_index][class_index])
#            for class_index in top_indices]
#    line = sorted(line, key=lambda p: -p[1])
#    yield video_ids[video_index].decode('utf-8') + "," + " ".join("%i %f" % pair
#                                                  for pair in line) + "\n"
    raw_label = numpy.nonzero(labels[video_index])[0]
    raw_id = video_ids[video_index]
    example = tf.train.Example(features=tf.train.Features(feature={
      #"labels":tf.train.Feature(int64_list=tf.train.Int64List(value=raw_label)),
      "video_id":tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_id])),
      "labels":tf.train.Feature(int64_list=tf.train.Int64List(value=raw_label)),
      FLAGS.feature_name:tf.train.Feature(float_list=tf.train.FloatList(value=features[video_index]))
    }))

    #print "video_id:"
    #print video_ids[video_index]
    #print "raw_label:"
    #print raw_label
    #print "saving feature..."
    #numpy.save("/mnt/share4/tangluming/feature_new/feature_new_"+str(video_index)+".npy",features[video_index])

    yield example.SerializeToString()


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, unused_labels, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch=True,
                            enqueue_many=True))
    return video_id_batch, video_batch, num_frames_batch, unused_labels

def inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k):
  #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess, gfile.Open(out_file_location, "w+") as out_file:
  
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess, tf.python_io.TFRecordWriter(out_file_location) as out_file:
 
    video_id_batch, video_batch, num_frames_batch, labels_batch = get_input_data_tensors(reader, data_pattern, batch_size)
	
    if FLAGS.checkpoint_name == "":
      latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    else:
      latest_checkpoint = FLAGS.train_dir+"model.ckpt-"+FLAGS.checkpoint_name
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % train_dir)
    else:
      meta_graph_location = latest_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)
    input_tensor = tf.get_collection("input_batch_raw")[0]
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    #keep_prob = tf.get_collection("keep_prob")[0]
    features = tf.get_collection("features")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()
    #out_file.write("VideoId,LabelConfidencePairs\n")
    
    #tang = 0

    try:
      while not coord.should_stop():
          video_id_batch_val, video_batch_val,num_frames_batch_val,labels_batch_val = sess.run([video_id_batch, video_batch, num_frames_batch, labels_batch])
          predictions_val, features_val= sess.run([predictions_tensor,features], feed_dict={input_tensor: video_batch_val, num_frames_tensor: num_frames_batch_val})
          now = time.time()
          num_examples_processed += len(video_batch_val)
          num_classes = predictions_val.shape[1]
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
          #for line in format_lines(video_id_batch_val, predictions_val, top_k):
          #print "tang info:"
          
          #print "id_batch"
          #print video_id_batch_val
          #print type(video_id_batch_val)
          #print numpy.shape(video_id_batch_val)
          
          #print "feature_batch"
          #print features_val
          #print type(features_val)
          #print numpy.shape(features_val)
          
          #print "label_batch"
          #print labels_batch_val
          #print type(labels_batch_val)
          #print numpy.shape(labels_batch_val)
          #print numpy.nonzero(labels_batch_val)
          for sample in format_lines(video_id_batch_val, features_val, labels_batch_val):
            out_file.write(sample)
            #tang+=1
          #out_file.flush()

          #if tang>10000:
          #  break


    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  inference(reader, FLAGS.train_dir, FLAGS.input_data_pattern,
    FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()
