import scipy.sparse as sp
import scipy.io
import inspect
import tensorflow as tf
from preprocessing import preprocess_graph, sparse_to_tuple

flags = tf.app.flags
FLAGS = flags.FLAGS



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# import scipy.io as sio
# import os, sys
# def see_MATLAB_headers():
    # for f in os.listdir('data/'):
        # if 'Flickr' not in f:
            # continue
        # print '---------- {} ----------'.format(f)
        # data = sio.loadmat('data/{}'.format(f))
        # for k in data:
            # print k
        # print
# see_MATLAB_headers()

# ---------- acm_test_final.mat ----------
# ---------- BlogCatalog.mat ----------
# __header__
# __version__
# __globals__
# Network
# Label
# Attributes
# Class
# ---------- Amazon.mat ----------
# ---------- Disney.mat ----------
# ---------- Enron.mat ----------
# __header__
# __version__
# __globals__
# X
# A
# gnd
# ---------- Flickr1.mat ----------
# ---------- Flickr2.mat ----------
# Network
# __globals__
# __header__
# Label
# Attributes
# __version__
# Class
# ---------- Flickr3.mat ----------
# Network
# __globals__
# __header__
# Label
# Attributes

def wrapper(f):
    if f in ('acm_test_final', 'BlogCatalog'):
        return ('Label', 'Attributes', 'Network')
    elif f in ('Amazon', 'Disney', 'Enron'):
        return ('gnd', 'X', 'A')
    elif f in ('Flickr1', 'Flickr2', 'Flickr3'):
        return ('Label', 'Attributes', 'Network')
    elif 'node-capture' in f:
        return ('Label', 'Attributes', 'Network')

def load_data(data_source):
    data = scipy.io.loadmat("data/{}.mat".format(data_source))
    keys = wrapper(data_source)
    labels = data[keys[0]]
    attributes = sp.csr_matrix(data[keys[1]])
    network = sp.lil_matrix(data[keys[2]])

    return network, attributes, labels

def format_data(data_source):

    adj, features, labels = load_data(data_source)

    # Store original adjacency matrix (without diagonal entries) for later
    # adj_orig = adj
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()
    # adj = adj_orig

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, adj_norm, adj_label, features, labels]
    feas = {}
    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]]
        item_name = retrieve_name(item)
        feas[item_name] = item

    return feas

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var and "item" not in var_name][0]
