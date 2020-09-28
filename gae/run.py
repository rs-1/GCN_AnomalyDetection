import os
import csv
import time
import itertools
import traceback
import sys
import tensorflow as tf
import numpy as np
from anomaly_detection import AnomalyDetectionRunner


SLURM_JOB_ID = os.getenv('SLURM_JOB_ID', '')


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 16, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
# flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 100, 'number of iterations.')


# '''
# We did not set any seed when we conducted the experiments described in the paper;
# We set a seed here to steadily reveal better performance of ARGA
# '''
# seed = 7
# np.random.seed(seed)
# tf.set_random_seed(seed)

model = 'gcn_ae'  # 'arga_ae' or 'arga_vae'
task = 'anomaly_detection'
data_list = [
        sys.argv[1],
        # 'acm_test_final',
        # 'BlogCatalog',
        # 'Flickr1',
        ]
alphas = [i/10.0 for i in range(0,10+1)]
NUM_TO_AVG = 1

def T(start):
    return int(time.time() - start)

def get_aucs(data_list):

    START = time.time()
    FILE = 'output/{}-{}-epochs-{}-avg.csv'.format(SLURM_JOB_ID, FLAGS.iterations, NUM_TO_AVG)

    settings = {'data_name': None, 'alpha': None, 'iterations' : FLAGS.iterations, 'model' : model}

    with open(FILE, 'a') as f:
        w = csv.writer(f)
        for dataname, alpha in itertools.product(data_list, alphas):
            settings['data_name'] = dataname
            settings['alpha'] = alpha
            for _ in range(NUM_TO_AVG):
                runner = AnomalyDetectionRunner(settings)
                exc_info = None
                try:
                    r = runner.erun()
                    r.append(T(START))
                    w.writerow(r)
                    f.flush()
                except Exception as e:
                    exc_info = sys.exc_info()
                finally:
                    if exc_info:
                        traceback.print_exception(*exc_info)
                        del exc_info

# TODO
# reduce iterations/epochs to 100, not much change anyway
# average out aucs multiple runs (10)
# accelerate with gpu for acm_test_final
# use multiple cores?
# modify anomaly_detection to output multiple runs to files

get_aucs(data_list)

'''
for dataname in data_list:
    settings = {'data_name': dataname, 'iterations' : FLAGS.iterations, 'model' : model}
    runner = AnomalyDetectionRunner(settings)
    exc_info = None
    try:
        print '---------------------------------------- ' + dataname + ' ----------------------------------------'
        print FLAGS.__flags
        runner.erun()
        print
        print
    except Exception as e:
        exc_info = sys.exc_info()
    finally:
        if exc_info:
            traceback.print_exception(*exc_info)
            del exc_info
'''
