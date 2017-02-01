import numpy as np
import tensorflow as tf
import quaternion_ops as qops


def create_move_map(n_beads, dx):
    '''Inputs:
    -n_beads: int, number of beads in the model
    -dx: float, magnitude of each rotation step, in radians. This will be used during each move in the folding trajectory.'''
    move_map = {}
    template = np.zeros(n_beads)
    move_counter = 0
    for i in range(2,n_beads):
        for j in [-dx, dx]:
            move = np.copy(template)
            move[i] = j
            move_map[str(move_counter)] = move
            move_counter += 1
    return move_map


def rotate(x, rot):
    '''x: a tuple of (x,y,z) coordinates, assuming (0,0) is the axis of rotation.
       rot: rotation quaternion, a length-4 numpy array.
       Returns a rotated point in Cartesian space.'''
    x = np.array([0, x[0], x[1], x[2]])
    rotated = qops.multiply(qops.inverse(rot), qops.multiply(x, rot))
    return np.array([rotated[1], rotated[2], rotated[3]])


def initialize_coords(x):
    coords = [None] * len(x)
    coords[0] = np.array([0,0,0])
    dx = np.cos(35 * np.pi / 180.0)
    dy = np.sin(35 * np.pi / 180.0)
    dz = 0
    vector = np.array([dx, dy, dz])
    for i in range(1,len(x)):
        coords[i] = coords[i-1] + vector
        vector[1] *= -1
    return coords
            

def compute_cartesian_coords(x):
    '''"x" stores the rotation information of each angle. This will map these rotations to Cartesian coordinates.'''
    initial = initialize_coords(x)
    initial = np.array(initial).astype(float)
    for i in range(1,len(x)):
        temp_origin = initial[i] # Pick out the rotation origin 
        temp_recenter = initial - temp_origin # Recenter the entire protein around that origin temporarily
        rotation_axis = (initial[i] - initial[i-1]) / np.linalg.norm(initial[i] - initial[i-1])
        for j in range(i+1, len(temp_recenter)):
            qrot = qops.rotation_to_q(rotation_axis, x[i+1])
            temp_recenter[j] = rotate(temp_recenter[j], qrot)
        initial = temp_recenter + temp_origin # Restore actual coordinates to all points
    return initial


def compute_energy(pos):
    '''Computes interaction energy between amino acids. This is simply equal to the inverse of the Euclidian distance between two amino acids for simplicity.'''
    if len(pos) == 0: # Illegal conformation represented by an empty positions list
        return float('Inf')
    E = np.zeros((len(pos), len(pos)))
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if np.sum((np.array(pos[i]) - np.array(pos[j])) ** 2) ** 0.5 > 0:
                energy = 1 / np.sum((np.array(pos[i]) - np.array(pos[j])) ** 2) ** 0.5    
                E[i][j] = energy
            else:
                return float('Inf')
    return np.sum(E)  


def full_forward(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['W1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['W2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['W3']) + biases['b3']
    return out_layer


def make_move(inputs, outputs, move_map):
    probs = np.exp(outputs) / np.sum(np.exp(outputs))
    selection = np.random.choice(range(len(probs)), p=probs)
    move = move_map[str(selection)]   
    new_inputs = inputs + move
    E = compute_energy(compute_cartesian_coords(new_inputs))
    label = np.zeros(len(move_map.keys()))
    label[selection] = 1
    return new_inputs, E, label


def discount_rewards(rewards, discount_factor, parity):
    for i in range(len(rewards)):
        rewards[len(rewards) - i - 1] *= discount_factor ** i
    return np.array([np.array(rewards) * parity])


# Initialize network
all_layers = [9, 9, 9, 7 * 2]
weights = {}
weight_scale = 1e-3
biases = {}
dx = 5 * np.pi / 180
move_map = create_move_map(all_layers[0], dx)
learning_rate = 1e-2
discount_factor = 0.99
n_epochs = 100000
for i in range(len(all_layers) - 1):
    weights['W%d'%(i+1)] = tf.Variable(tf.random_normal([all_layers[i], all_layers[i+1]], stddev=weight_scale))
    biases['b%d'%(i+1)] = tf.Variable(tf.zeros([1, all_layers[i+1]]))

x = tf.placeholder("float", [None, all_layers[0]], name='x')
labels = tf.placeholder("float", [None, all_layers[-1]], name='labels')
Rewards = tf.placeholder("float", [1, None], name='rewards')
pred = full_forward(x, weights, biases)
loss = tf.reduce_mean(tf.mul(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels),Rewards))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.initialize_all_variables()

step_counter = 0
inputs = np.zeros(all_layers[0])
E_cache = [compute_energy(compute_cartesian_coords(inputs))]

print 'Initial energy %f'%(E_cache[0])
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_epochs):
        print 'Epoch %d'%(i)
        step_counter = 0
        rewards = []
        path_E = [] # A record of all energies along a particular path
        X = [] # X data to feed into model
        Y = [] # Y data to feed into model
        inputs = np.ones(all_layers[0]) # Input dihedrals. Should be 0, but that would give 0 outputs and mess up training, so pad with 1
        X.append(inputs)
        while True:
            out = sess.run(pred, feed_dict={x:np.array([inputs])})[0]
            inputs, E, label = make_move(inputs - 1.0, out, move_map) # Subtract 1 padding to compute the actual energy
            path_E.append(E)
            Y.append(label)
            rewards.append(1.0)
            step_counter += 1
            if step_counter > 500:
                path_E = np.array(path_E)
                good_energy_locs = np.where(path_E > np.max(np.array(E_cache)))[0]
                if len(good_energy_locs) > 0:
                    best_energy_loc = np.where(path_E == np.max(path_E))[0][0]
                    print 'Found new energy basins!'
                    print 'Best energy %f is at frame %d. Truncating at this frame.'%(path_E[best_energy_loc], best_energy_loc)
                    E_cache.append(path_E[best_energy_loc])
                    rewards = discount_rewards(rewards[:best_energy_loc], discount_factor, 1)
                    X = X[:best_energy_loc]
                    Y = Y[:best_energy_loc]
                    sess.run([optimizer], feed_dict={x:np.array(X), labels: np.array(Y), Rewards: rewards})
                else:
                    rewards = discount_rewards(rewards, discount_factor, -0.2)
                    sess.run([optimizer], feed_dict={x:np.array(X), labels: np.array(Y), Rewards: rewards})
                break
            else:
                X.append(inputs)
             
