import tensorflow as tf
import numpy as np
import glob

# Match all the files in the feature directory
file_list = glob.glob('/mnt/share2/youtube-8m_data/features_video/train*.tfrecord')

n = 4716
cnt_j = np.zeros(n)
cnt_ij = np.zeros([n, n])
total = len(file_list)
cur = 0

for rec in file_list:
  cur = cur + 1
  print "- Processing [%d/%d]" % (cur, total)
  for example in tf.python_io.tf_record_iterator(rec):
    info = tf.train.Example.FromString(example)
    labels = info.features.feature['labels'].int64_list.value
    sz = len(labels)
    for i in range(sz):
      cnt_j[labels[i]] += 1
      for j in range(i + 1, sz, 1):
        cnt_ij[labels[i]][labels[j]] += 1
        cnt_ij[labels[j]][labels[i]] += 1

np.save("/mnt/lustre/share/dengby/video-pred/raw_stats_ij.npy", cnt_ij)
np.save("/mnt/lustre/share/dengby/video-pred/raw_stats_j.npy", cnt_j)
