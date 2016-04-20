import tensorflow as tf
import pickle
from data_structures import *
import numpy as np

managers = pickle.load(open("data/managers.p", 'r'))

train_man = managers['train_man']
val_man = managers['val_man']
test_man = managers['test_man']

print(train_man.molecules)

print(train_man.molecules[0].get_atoms())

N_atom_feat = 62
max_deg = 5
fp_sz = 512
bool_fp_partition = False

N_radius = 1#1
flow_layer_depths = [N_atom_feat, 200]
bv_reg_type = "L2"
bv_weight_reg = 1e-2

standard_hidden_dim = [1000]

conv_hid_weight_scale = 1e-2
conv_w_weight_scale = 1e-4
standard_w_weight_scale = 1e-2
learning_rate = 1e-3#1e-5#0.0001#0.001

N_bond_types = 17 # Excluding non bonds, but including self bond
bool_bond_flow = True
bond_flow_weight_scale = 1e-2
bond_flow_reg = 1e-1

atom_to_sparse_reg_type = "L2"
atom_to_sparse_reg = 5e-2

standard_w_weight_reg = 5e-1
regressor_w_weight_reg = 5e-1

#bond_flow_center = np.array([0] + N_bond_types*[1])
# bond_flow_center = np.array([0] + N_bond_types*[1])

N_epochs = 100

# Placeholders
atoms_ph = tf.placeholder(tf.float32, shape=[None,N_atom_feat], name='atoms')
adj_mat_ph = tf.placeholder(tf.float32, shape=[None,None], name='adj_mat')
type_adj_ph = tf.placeholder(tf.int32, shape=[None,None], name='type_adj')
deg_list_ph = tf.placeholder(tf.int32, shape=[None], name='deg_list')
N_atoms_ph = tf.placeholder(tf.int32, name='N_atoms')
target_ph = tf.placeholder(tf.float32, name='targets')

# A few more calculations
if bool_fp_partition:
    fp_real_sz = (N_radius+1)*fp_sz
else:
    fp_real_sz = fp_sz
    
"""
from tensorflow.python.ops.gen_math_ops import _range as _tf_range


def tf_range(start, limit=None, delta=1, name="range"):
    return _tf_range(start, limit, delta, name=name)
"""

#@tf.RegisterGradient("DynamicPartition")
#def _DynamicPartitionGrads(op, *grads):
#  """#Gradients for DynamicPartition."""
#  data = op.inputs[0]
#  indices = op.inputs[1]
#  num_partitions = op.get_attr("num_partitions")
#
#  prefix_shape = tf.shape(indices)
#  original_indices = tf.reshape(tf.range(tf.reduce_prod(prefix_shape)), prefix_shape)
#  partitioned_indices = tf.dynamic_partition(original_indices, indices, num_partitions)
#  reconstructed = tf.dynamic_stitch(partitioned_indices, grads)
#  reconstructed = tf.reshape(reconstructed, tf.shape(data))
#  return [reconstructed, None]

def init():
    # Bondvolutional parameters
    bv_params = {}
    
    for layer in range(1,N_radius+1):
        for deg in range(1,max_deg+1):
            flow_depth_prev = flow_layer_depths[layer-1]
            flow_depth = flow_layer_depths[layer]

            # Initialize non zero parameters
            non_zeros = 0.0 +  tf.Variable(tf.random_normal([N_bond_types, flow_depth_prev, flow_depth], stddev=bond_flow_weight_scale))

            # Concat zero parameter in front of non_zero parameters
            bv_params['A_flow'+str(layer)+'_'+str(deg)] = tf.concat(0, [np.zeros([1, flow_depth_prev, flow_depth]), non_zeros])

            # Create biases
            bv_params['b_flow'+str(layer)+'_'+str(deg)] = tf.Variable(tf.zeros([1, 1, flow_depth]))
            
    # Convolutional Fingerprint translation parameters
    net_params = {}
    
    net_params['cFW'+str(0)] = tf.Variable(tf.random_normal([N_atom_feat, fp_sz], stddev=conv_w_weight_scale),
                                           name="cFW"+str(deg))
    net_params['cFb'+str(0)] = tf.Variable(tf.zeros([fp_sz]),
                                           name="cFb"+str(deg))

    for layer in range(1, N_radius+1):
        flow_depth = flow_layer_depths[layer]
        
        net_params['cFW'+str(layer)] = tf.Variable(tf.random_normal([flow_depth, fp_sz], stddev=conv_w_weight_scale),
                                                   name="cFW"+str(deg))
        net_params['cFb'+str(layer)] = tf.Variable(tf.zeros([fp_sz]),
                                                   name="cFb"+str(deg))

    # Standard (vanilla) layer 
    net_params['sDW1'] = tf.Variable(tf.random_normal([fp_real_sz, standard_hidden_dim[0]], stddev=standard_w_weight_scale),
                                     name="sDW1")
    net_params['sDb1'] = tf.Variable(tf.zeros([standard_hidden_dim[0]]),
                                     name='sDb1')
    
    #net_params['sDW2'] = tf.Variable(tf.random_normal([fp_real_sz, standard_hidden_dim[1]], stddev=standard_w_weight_scale),
#                                     name="sDW2")
    #net_params['sDb2'] = tf.Variable(tf.zeros([standard_hidden_dim[1]]),
#                                     name='sDb2')
    
    # Final regressor layer
    net_params['sRW'] = tf.Variable(tf.random_normal([standard_hidden_dim[0],1], stddev=standard_w_weight_scale),
                                    name='sRW')
    net_params['sRb'] = tf.Variable(tf.constant(0.0),
                                    name='sRb')
    
    return net_params, bv_params

def affine(x, W, b):
    return tf.add(tf.matmul(x, W), b)

def bond_conv_layer(activated_atoms, bv_params, layer):
    flow_depth = flow_layer_depths[layer]
    
    next_activated_atoms = tf.zeros(tf.pack([N_atoms_ph, flow_depth]))

    for deg in range(1, 6):
        indices = tf.sub(deg_list_ph, tf.constant(1,dtype=tf.int32))
        flow_param = bv_params['A_flow'+str(layer)+'_'+str(deg)]
        flow_map = tf.gather(flow_param, type_adj_ph)

        multiples = tf.pack([N_atoms_ph, 1, 1])
        activated_atoms_dim = tf.expand_dims(tf.tile(tf.expand_dims(activated_atoms, 0), multiples), 2)

        adj_mul = tf.batch_matmul(activated_atoms_dim, flow_map)
        adj_mul = tf.squeeze(adj_mul, [2])

        deg_mask = tf.to_float(tf.equal(deg_list_ph, deg))

        multiples = tf.pack([1, N_atoms_ph, flow_depth])
        deg_list_dim = tf.tile(tf.expand_dims(tf.expand_dims(deg_mask, 1), 1), multiples)

        multiples = tf.pack([N_atoms_ph, N_atoms_ph, 1])
        biases = tf.tile(bv_params['b_flow'+str(layer)+'_'+str(deg)], multiples)
        filtered_atoms = tf.add(tf.mul(adj_mul, deg_list_dim), biases)

        next_activated_atoms = next_activated_atoms + tf.reduce_sum(filtered_atoms, 1)
        
    next_activated_atoms = tf.nn.relu(next_activated_atoms)
    return next_activated_atoms

def sum_neigh(atoms, aux_params, layer):
    if bool_bond_flow:
        # Replace the bond with the bond weights
        flow_depth = flow_layer_depths[layer-1]
        flow_map = []
        for k in range(flow_depth):
            #tiled_type_adj = tf.tile(tf.expand_dims(type_adj_ph, 2), [1,1,flow_depth])
            #print(tiled_type_adj)
            #flow_mat = tf.gather(aux_params['A_flow'+str(layer)], tiled_type_adj)
            #print(flow_mat)
            flow_map.append(tf.gather(aux_params['A_flow'+str(layer)][:,k], type_adj_ph))
            #flow_mat = tf.Print(flow_mat, [flow_mat])
        atom_mult_neigh = tf.matmul(flow_mat, atoms)
        atom_sum_neigh = tf.add(atom_mult_neigh, atoms)
    else:
        atom_mult_neigh = tf.matmul(adj_mat_ph, atoms)
        atom_sum_neigh = tf.add(atom_mult_neigh, atoms)
        
    return atom_sum_neigh

def mol_conv_layer(atoms, cH_params, aux_params, layer):
    #Sum all neighbors using adjacency matrix
    atom_sum_neigh = sum_neigh(atoms, aux_params, layer)

    # Partition the atom matrix by degree of atoms
    # THIS CREATES PROBLEMS WITH GRADIENTS. NEED TO USE SLICING
    indices = tf.sub(deg_list_ph, tf.constant(1,dtype=tf.int32))
    
    atom_partitions = tf.dynamic_partition(atom_sum_neigh, indices, max_deg)

    # Get collection of modified atom features
    new_rel_atoms_collection = []
    for deg in range(1,6):
        # Obtain relevant atoms for this degree
        rel_atoms = atom_partitions[deg-1]

        # Apply hidden affine to relevant atoms and append
        if bool_separate_conv_depths:
            out = affine(rel_atoms, cH_params['W'+str(deg)+'_'+str(layer)], cH_params['b'+str(deg)+'_'+str(layer)])
        else:
            out = affine(rel_atoms, cH_params['W'+str(deg)], cH_params['b'+str(deg)])
        new_rel_atoms_collection.append(out)

    # Combine all atoms back into the list
    # NOTE: FOR NOW USE CONCATENATION. MEANS WE CANNOT USE ARBITARY deg_list ORDER
    hidden_atoms = tf.concat(0, new_rel_atoms_collection)

    # Apply relu
    activated_atoms = tf.nn.relu(hidden_atoms)

    return activated_atoms

def get_sparse_rep(atoms, layer_params):
    # Apply affine
    scores = affine(atoms, layer_params['cFW'], layer_params['cFb'])

    # Sparisify
    sparse_rep = tf.reduce_sum(tf.nn.softmax(scores), 0, keep_dims=True)

    # Add to fingerprint
    return sparse_rep#= fp.append(sparse_rep)

def mcnn_pred(atoms):
    # Initialize fingerprint
    fp_list = []

    activated_atoms = atoms
                        
    # Initial sparse representation
    layer_params = {'cFW': net_params['cFW'+str(0)], 'cFb': net_params['cFb'+str(0)]}
    fp = get_sparse_rep(activated_atoms, layer_params)
    fp_list.append(fp)

    for layer in range(1, N_radius+1):
        # Apply Convolutional layers
 
        activated_atoms = bond_conv_layer(activated_atoms, bv_params, layer)

        layer_params = {'cFW': net_params['cFW'+str(layer)], 'cFb': net_params['cFb'+str(layer)]}        
        fp = get_sparse_rep(activated_atoms, layer_params)
        fp_list.append(fp)
        
    if bool_fp_partition:
        fp = tf.concat(1, fp_list)
    else:
        fp = tf.reduce_sum(tf.concat(0,fp_list),0, keep_dims=True)

    # Calculate hidden affine on fingerprint and apply relu
    hidden = affine(fp, net_params['sDW1'], net_params['sDb1'])
    hidden = tf.nn.relu(hidden)

    #hidden = affine(fp, net_params['sDW2'], net_params['sDb2'])
    #hidden = tf.nn.relu(hidden)
    
    # Final regression
    score = affine(hidden, net_params['sRW'], net_params['sRb'])

    return score

def mcnn_loss(atoms, target):
    score = mcnn_pred(atoms)

    # Apply L2 loss
    return tf.nn.l2_loss(tf.sub(score, target)) + hidden_reg() + standard_reg()

def unreg_mcnn_loss(atoms, target):
    score = mcnn_pred(atoms)
        
    # Apply L2 loss
    return tf.nn.l2_loss(tf.sub(score, target))

def mcnn_error(atoms, target):
    score = mcnn_pred(atoms)
    
    # Apply L1 loss
    return tf.abs(tf.sub(score, target))

def hidden_reg():
    reg = tf.constant(0.0, dtype=tf.float32)

    for layer in range(1, N_radius+1):
        for deg in range(1,6):
            if bv_reg_type == "L1":
                reg = reg + bv_weight_reg * tf.reduce_mean(tf.abs(bv_params['A_flow'+str(layer)+'_'+str(deg)]))
            else:
                reg = reg + bv_weight_reg * tf.reduce_mean(tf.square(bv_params['A_flow'+str(layer)+'_'+str(deg)]))

    for layer in range(0, N_radius+1):
        if atom_to_sparse_reg_type == "L2":
            reg = reg + atom_to_sparse_reg * tf.reduce_mean(tf.square(net_params['cFW'+str(layer)]))
        else:
            reg = reg + atom_to_sparse_reg * tf.reduce_mean(tf.abs(net_params['cFW'+str(layer)]))            
    return reg

def standard_reg():
    reg = tf.constant(0.0, dtype=tf.float32)
    reg = reg + standard_w_weight_reg * tf.reduce_mean(tf.square(net_params['sDW1']))
    #reg = reg + standard_w_weight_reg * tf.reduce_mean(tf.square(net_params['sDW2']))    
    reg = reg + regressor_w_weight_reg * tf.reduce_mean(tf.square(net_params['sRW']))

    return reg

# Initialize Parameters and save dictioanry items
net_params, bv_params = init()

# Compile loss instance function for a particular molecule
loss_instance = mcnn_loss(atoms_ph, target_ph)
error_instance = mcnn_error(atoms_ph, target_ph)

# Initialize optimizer class
#optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.98, momentum=0.9, epsilon=1e-10)
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.98, beta2=0.9995, epsilon=1e-10)

# Create optimizer step tensorflow variable
global_step = tf.Variable(0, name='global_step', trainable=False)
# Compile function for minimizing a loss instance. 
train_op = optimizer.minimize(loss_instance, global_step=global_step)

# Compile function for initializing all variables
init_op = tf.initialize_all_variables()

N_train = train_man.get_N_molecules()
print(N_train)

def gen_data_set(data_man):

    data_set = {}

    atoms_list = []
    adj_mat_list = []
    type_adj_list = []    
    deg_list_list = []
    N_atoms_list = []
    target_list = []

    data_set['atoms_list'] = atoms_list
    data_set['adj_mat_list'] = adj_mat_list
    data_set['type_adj_list'] = type_adj_list
    data_set['deg_list_list'] = deg_list_list
    data_set['N_atoms_list'] = N_atoms_list
    data_set['target_list'] = target_list

    N_set = data_man.get_N_molecules()
    print(N_set)

    for k in range(N_set):
        # Get molecule data
        atoms = np.array(data_man.molecules[k].get_atoms())
        N_atoms = atoms.shape[0]
        atoms_list.append(atoms)
        
        adj_mat_list.append(np.array(data_man.molecules[k].get_adj_mat()))
        #print(k, data_man.molecules[k])
        type_adj_list.append(data_man.molecules[k].get_type_adj())
        deg_list = np.array(data_man.molecules[k].get_deg_list())

        # Create slice lists
        slice_begin_list = []
        slice_size_list = []

        print(data_man.molecules[k].get_type_adj())

        cur_deg = 1
        slice_begin_list.append(0)
        for i in range(N_atoms):
            while deg_list[i] != cur_deg:
                slice_size = i-slice_begin_list[cur_deg-1]
                slice_size_list.append(slice_size)
                cur_deg += 1
                slice_begin_list.append(i)

        deg_list_list.append(deg_list)
        N_atoms_list.append(N_atoms)

        # Get target
        target_list.append(data_man.targets[k])

    return data_set

def sess_compute_err(sess, data_set, bool_verbose=False):
    atoms_list = data_set['atoms_list']
    adj_mat_list = data_set['adj_mat_list']
    type_adj_list = data_set['type_adj_list']
    deg_list_list = data_set['deg_list_list']
    N_atoms_list = data_set['N_atoms_list']
    target_list = data_set['target_list']

    N_set = len(atoms_list)

    # Compute initial training errorp
    total_err = 0.0
    total_loss = 0.0
    err_vec = []
    for k in range(N_set):
        if bool_bond_flow:
            err, loss = sess.run([error_instance, loss_instance], feed_dict={atoms_ph:atoms_list[k], 
                                                       type_adj_ph:type_adj_list[k],
                                                       deg_list_ph:deg_list_list[k],
                                                       N_atoms_ph:N_atoms_list[k],
                                                       target_ph:target_list[k]})
        else:
            err, loss = sess.run([error_instance, loss_instance], feed_dict={atoms_ph:atoms_list[k], 
                                                       adj_mat_ph:adj_mat_list[k],
                                                       deg_list_ph:deg_list_list[k],
                                                       N_atoms_ph:N_atoms_list[k],
                                                       target_ph:target_list[k]})
        
        total_loss += loss
        total_err += err[0]
        # store vector or errors
        err_vec.append(err)
    # Get average error
    avg_err = total_err / N_set
    avg_loss = total_loss / N_set
    rmse = np.sqrt(np.sum(np.array(err_vec)**2)/N_set)
    if bool_verbose:
        print(err_vec)

    return avg_err, avg_loss, rmse

def sess_optimize(sess, train_set, val_set):
    # Perform the optimization
    
    atoms_list = train_set['atoms_list']
    adj_mat_list = train_set['adj_mat_list']
    type_adj_list = train_set['type_adj_list']
    deg_list_list = train_set['deg_list_list']
    N_atoms_list =train_set['N_atoms_list']
    target_list = train_set['target_list']

    for i in range(N_epochs):
        print("Epoch: "+str(i))
        train_err, train_loss, train_rmse = sess_compute_err(sess, train_set)
        val_err, val_loss, val_rmse = sess_compute_err(sess, val_set)
        print("average training error: %f" % train_err)
        print("average validation error: %f" % val_err)
        print("average training loss: %f" % train_loss)
        print("average validation loss: %f" % val_loss)
        print("average training rmse: %f" % train_rmse)
        print("average validation rmse: %f" % val_rmse)

        #tf.Print(aux_params['A_flow'], [aux_params['A_flow']], message='flow ')

        for k in range(N_train):
            if bool_bond_flow:
                sess.run([train_op], feed_dict={atoms_ph:atoms_list[k], 
                                                type_adj_ph:type_adj_list[k],
                                                deg_list_ph:deg_list_list[k],
                                                N_atoms_ph:N_atoms_list[k],
                                                target_ph:target_list[k]})
            else:
                sess.run([train_op], feed_dict={atoms_ph:atoms_list[k], 
                                                adj_mat_ph:adj_mat_list[k],
                                                deg_list_ph:deg_list_list[k],
                                                N_atoms_ph:N_atoms_list[k],
                                                target_ph:target_list[k]})

train_set = gen_data_set(train_man)
val_set = gen_data_set(val_man)
test_set = gen_data_set(test_man)

print("Beginning TensorFlow")

with tf.Session() as sess:
    sess.run(init_op)

    init_err, init_loss, init_rmse = sess_compute_err(sess, train_set)
    print("initial average error: %f" % init_err)
    print("initial average loss: %f" % init_loss)
    print("initial average rmse: %f" % init_rmse)        
    sess_optimize(sess, train_set, val_set)

    test_err, test_loss, test_rmse = sess_compute_err(sess, test_set)
    print("average test error: %f" % test_err)
    print("average test loss: %f" % test_loss)
    print("average test rmse: %f" % test_rmse)    

    sess.close()
    print("done")
