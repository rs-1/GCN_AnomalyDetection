import csv
import time
import itertools
import traceback
import sys
import tensorflow as tf
import numpy as np
from anomaly_detection import AnomalyDetectionRunner


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

data_list = [
        'acm_test_final',
        #'BlogCatalog',
        #'Flickr1',
        ]
alphas = [i/10.0 for i in range(0,10+1)]
model = 'gcn_ae'  # 'arga_ae' or 'arga_vae'
task = 'anomaly_detection'

def get_aucs_per_dataname_per_alpha(dataname, alpha, num):
    settings = {'data_name': dataname, 'alpha': alpha, 'iterations' : FLAGS.iterations, 'model' : model}
    results = []
    for _ in range(num):
        runner = AnomalyDetectionRunner(settings)
        exc_info = None
        try:
            r = runner.erun()
            results.append(r)
        except Exception as e:
            exc_info = sys.exc_info()
        finally:
            if exc_info:
                traceback.print_exception(*exc_info)
                del exc_info
    return results

def get_aucs(data_list):
    T = time.time
    num = 10
    for dataname, alpha in itertools.product(data_list, alphas):
        try:
            results = get_aucs_per_dataname_per_alpha(dataname, alpha, num)
        except Exception as e:
            results = []
        finally:
            with open('output/{}-{}-{}.csv'.format(int(T()), dataname, alpha), 'wb') as f:
                out = csv.writer(f)
                out.writerows(results)

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
