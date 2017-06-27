'''
  argv[1] : stats file
  argv[2] : prediction_0
  argv[3] : target_file
'''
import sys
import numpy as np
import scipy as sp
import scipy.io

mat = sp.io.loadmat(sys.argv[1])
mx = mat['mx'].reshape(4716)
mn = mat['mn'].reshape(4716)
rg = mat['range'].reshape(4716)
wij = mat['wij']

print "=> Loading original predictions ..."
prob_0 = np.load(sys.argv[2])
print "=> Load Done!"
total = prob_0.shape[0]
top_k = 20

with open(sys.argv[3], "w+") as out_file: 
  out_file.write("VideoId,LabelConfidencePairs\n")
  for i in range(total):
    print "-- On processing [%d/%d]" % (i + 1, total)
    prob = prob_0[i][:-1]
    index = int(prob_0[i][-1])
    prob_1 = np.matmul(wij, prob.T)
    prob_1 = (prob_1 - mn) / rg
    top_indices = np.argpartition(prob_1, -top_k)[-top_k:]
    line = [(class_index, prob_1[class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    out_file.write(str(index) + "," + " ".join("%i %f" % pair for pair in line) + "\n")  
    out_file.flush()
    
print "Done!"
