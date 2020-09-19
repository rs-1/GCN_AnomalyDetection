import traceback
import sys
import tensorflow as tf
import numpy as np
from anomaly_detection import AnomalyDetectionRunner


try:
    ALPHA = float(sys.argv[1])
except Exception as e:
    print 'ERROR: Bad argument for ALPHA parameter'
    sys.exit()


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
flags.DEFINE_integer('iterations', 300, 'number of iterations.')
flags.DEFINE_float('alpha', ALPHA, 'balance parameter')


# '''
# We did not set any seed when we conducted the experiments described in the paper;
# We set a seed here to steadily reveal better performance of ARGA
# '''
# seed = 7
# np.random.seed(seed)
# tf.set_random_seed(seed)

data_list = [
        'acm_test_final',
        'Amazon',
        'BlogCatalog',
        'Disney',
        'Enron',
        'Flickr1',
        'Flickr2',
        'Flickr3',
        ]

for dataname in data_list:
    model = 'gcn_ae'  # 'arga_ae' or 'arga_vae'
    task = 'anomaly_detection'
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
