import tensorflow as tf
from utils.nn import linearND
import math, sys, random, os
from optparse import OptionParser
import threading
from multiprocessing import Queue, Process
import numpy as np
from Queue import Empty
import time
import h5py
from itertools import chain
import os 
project_root = os.path.dirname(os.path.dirname(__file__))

NK = 10
NK0 = 5
report_interval = 1
max_save = 30
min_iterations = 1000

score_scale = 5.0
min_separation = 0.25

FP_len = 1024
FP_rad = 2

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path", default=os.path.join(project_root, 'data', 'reaxys_limit10.txt'))
parser.add_option("--h5", dest="h5_suffix", default=".h5")
parser.add_option("-m", "--save_dir", dest="save_path", default=os.path.join(project_root, 'models', 'example_model'))
parser.add_option("-b", "--batch", dest="batch_size", default=16384)
parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
parser.add_option("-d", "--depth", dest="depth", default=5)
parser.add_option("-l", "--max_norm", dest="max_norm", default=5.0)
parser.add_option("-u", "--device", dest="device", default="")
parser.add_option("--test", dest="test", default='')
parser.add_option("-v", "--verbose", dest="verbose_test", default=False)
parser.add_option("-c", "--checkpoint", dest="checkpoint", default="final")
parser.add_option("-s", "--saveint", dest="save_interval", default=0)
parser.add_option("-i", "--interactive", dest="interactive", default=False)
opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
max_norm = float(opts.max_norm)
test = opts.test
save_interval = int(opts.save_interval)
verbose_test = bool(opts.verbose_test)
interactive_mode = bool(opts.interactive)
h5_suffix = opts.h5_suffix

if '2048' in h5_suffix:
    FP_len = 2048

if interactive_mode:
    batch_size = 2 # keep it small

if not os.path.isdir(opts.save_path):
    os.mkdir(opts.save_path)

import rdkit.Chem.AllChem as AllChem
if 'counts' not in opts.save_path and 'uint8' not in opts.h5_suffix:
    # bool version
    def mol_to_fp(mol, radius=FP_rad, nBits=FP_len):
        if mol is None:
            return np.zeros((nBits,), dtype=np.float32)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, 
            useChirality=True), dtype=np.bool)
else:
    # uint8 version
    def mol_to_fp(mol, radius=FP_rad, nBits=FP_len, convFunc=np.array):
        if mol is None:
            return convFunc((nBits,), dtype=dtype)
        fp = AllChem.GetMorganFingerprint(mol, radius, useChirality=True) # uitnsparsevect
        fp_folded = np.zeros((nBits,), dtype=dtype)
        for k, v in fp.GetNonzeroElements().iteritems():
            fp_folded[k % nBits] += v 
        return convFunc(fp_folded)

def smi_to_fp(smi, radius=FP_rad, nBits=FP_len):
    if not smi:
        return np.zeros((nBits,), dtype=np.float32)
    return mol_to_fp(Chem.MolFromSmiles(smi), radius, nBits)

gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=opts.device)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    _input_mol = tf.placeholder(tf.float32, [batch_size*2, FP_len])
    sa_target = tf.placeholder(tf.float32, [batch_size*2,])

    q = tf.FIFOQueue(20, [tf.float32], shapes=[[batch_size*2, FP_len]]) # fixed size
    enqueue = q.enqueue(_input_mol)
    input_mol = q.dequeue()
    src_holder = [input_mol]

    input_mol.set_shape([batch_size*2, FP_len])

    mol_hiddens = tf.nn.relu(linearND(input_mol, hidden_size, scope="encoder0"))
    for d in xrange(1, depth):
        mol_hiddens = tf.nn.relu(linearND(mol_hiddens, hidden_size, scope="encoder%i"%d))

    score_sum = linearND(mol_hiddens, 1, scope="score_sum")
    score_sum = tf.squeeze(score_sum)
    score = 1.0 + (score_scale - 1.0) * tf.nn.sigmoid(score_sum)

    # For evaluation only - get SSE against a target
    sse = tf.reduce_sum(tf.square(score - sa_target))
    
    pm_one = tf.constant([-1, 1], dtype=tf.float32)
    reshape_score = tf.reshape(score, [batch_size, 2])
    reshape_score = tf.multiply(reshape_score, pm_one) # products minus reactants
    diff_score = tf.reduce_sum(reshape_score, axis=-1)

    # shifted ReLU loss (like hinge loss)
    # want to have positive diff score - min_separation > 0
    loss = tf.nn.relu(min_separation - diff_score)
    loss = tf.reduce_sum(loss)

    # For normal reaction-wise training
    _lr = tf.placeholder(tf.float32, [])
    optimizer = tf.train.AdamOptimizer(learning_rate=_lr)
    param_norm = tf.global_norm(tf.trainable_variables())
    grads_and_vars = optimizer.compute_gradients(loss / batch_size)
    grads, var = zip(*grads_and_vars)
    grad_norm = tf.global_norm(grads)
    new_grads, _ = tf.clip_by_global_norm(grads, max_norm)
    grads_and_vars = zip(new_grads, var)
    backprop = optimizer.apply_gradients(grads_and_vars)

    # For training if exact values known (unused)
    sse_grads_and_vars = optimizer.compute_gradients(sse / batch_size / 2.0)
    sse_grads, sse_var = zip(*sse_grads_and_vars)
    sse_grad_norm = tf.global_norm(sse_grads)
    sse_new_grads, _ = tf.clip_by_global_norm(sse_grads, max_norm)
    sse_grads_and_vars = zip(sse_new_grads, sse_var)
    sse_backprop = optimizer.apply_gradients(sse_grads_and_vars)

    tf.global_variables_initializer().run(session=session)
    size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size_func(v) for v in tf.trainable_variables())
    print "Model size: %dK" % (n/1000,)

    queue = Queue()


    def read_data_once(path, coord, frag='valid'):
        if os.path.isfile(path + '.pkl'):
            with open(path + '.pkl', 'r') as fid:
                data = pickle.load(fid)
        else:
            data = []
            with open(path, 'r') as f:
                for line in f:
                    rex, n, _id = line.strip("\r\n").split(' ')
                    r,p = rex.split('>>')
                    if ('.' in p) or (not p):
                        continue # do not allow multiple products or none
                    n = int(n)
                    for r_splt in r.split('.'):
                        if r_splt:
                            data.append((_id, n, r_splt, p))
            random.seed(123)
            random.shuffle(data)
            with open(path + '.pkl', 'w') as fid:
                data = pickle.dump(data, fid, -1)

        # h5py was generated post-shuffle
        f = h5py.File(path + h5_suffix, 'r')
        data_fps = f['data_fps']

        data_len = len(data)
        print('After splitting, %i total data entries' % data_len)
        if frag == 'train':
            data = data[:int(0.8 * data_len)]
            h5_offset = 0*2
            data_len = len(data)
            print('Taking pseudo-random 0.8 as training set (%i)' % data_len)
        elif frag == 'valid':
            data = data[int(0.8 * data_len):int(0.9 * data_len)]
            h5_offset = int(0.8 * data_len)*2
            data_len = len(data)
            print('Taking pseudo-random 0.1 as validation set (%i)' % data_len)
        elif frag == 'test':
            data = data[int(0.9 * data_len):]
            h5_offset = int(0.9 * data_len)*2
            data_len = len(data)
            print('Taking pseudo-random 0.1 as test set (%i)' % data_len)
        else:
            raise ValueError('Unknown data frag type')
        print('h5 offset: {}'.format(h5_offset))
        it = 0
        src_mols = np.zeros((batch_size*2, FP_len), dtype=np.float32)
        while it < data_len:

            # Try to get all FPs in one read (faster)
            if (it + batch_size) <= data_len:
                src_batch = list(chain.from_iterable((data[i][2], data[i][3]) for i in xrange(it, it + batch_size)))
                ids_batch = [data[i][0] for i in xrange(it, it + batch_size)]
                src_mols[:, :] = data_fps[h5_offset+2*it:h5_offset+2*(it+batch_size), :]
                it = it + batch_size
            # If we are at the end, do one-by-one)
            else:
                src_batch = []
                ids_batch = []
                for i in xrange(batch_size):
                    if it >= data_len:
                        src_batch.append(r)
                        src_batch.append(p)
                        ids_batch.append(_id)
                        src_mols[2*i:2*i+2, :] = np.zeros((2, FP_len))
                    else:
                        _id, n, r, p = data[it]
                        src_batch.append(r)
                        src_batch.append(p)
                        ids_batch.append(_id)
                        src_mols[2*i:2*i+2, :] = data_fps[h5_offset+2*it:h5_offset+2*it+2, :]
                    it = it + 1

            session.run(enqueue, feed_dict={_input_mol: src_mols})
            queue.put((ids_batch, src_batch))

        # Stop signal for testing
        queue.put((None, None))
        coord.request_stop()

    def read_data_master(path, coord):
        if not os.path.isfile(path + h5_suffix):
            quit('Need to run .h5 script first to get FPs')

        if os.path.isfile(path + '.pkl'):
            with open(path + '.pkl', 'r') as fid:
                data = pickle.load(fid)
        else:
            data = []
            with open(path, 'r') as f:
                for line in f:
                    rex, n, _id = line.strip("\r\n").split(' ')
                    r,p = rex.split('>>')
                    if ('.' in p) or (not p):
                        continue # do not allow multiple products or none
                    n = int(n)
                    for r_splt in r.split('.'):
                        if r_splt:
                            data.append((_id, n, r_splt, p))
            random.seed(123)
            random.shuffle(data)
            with open(path + '.pkl', 'w') as fid:
                data = pickle.dump(data, fid, -1)

        # h5py is post-shuffle
        f = h5py.File(path + h5_suffix, 'r')
        data_fps = f['data_fps']

        data_len = len(data)
        print('After splitting, %i total data entries' % data_len)
        print('...slicing data')
        data = data[:int(0.8 * data_len)]
        print('...NOT slicing h5 FP dataset, but defining offset (= 0)')  
        h5_offset = 0      

        data_len = len(data)
        print('Taking pseudo-random 0.8 for training (%i)' % data_len)
        
        it = 0; 
        src_mols = np.zeros((batch_size*2, FP_len), dtype=np.float32)
        while not coord.should_stop():

            # Try to get all FPs in one read (faster)
            if (it + batch_size) <= data_len:
                src_batch = list(chain.from_iterable((data[i][2], data[i][3]) for i in xrange(it, it + batch_size)))
                ids_batch = [data[i][0] for i in xrange(it, it + batch_size)]
                src_mols[:, :] = data_fps[h5_offset+2*it:h5_offset+2*(it+batch_size), :]
                it = (it + batch_size) % data_len

            # If we are at the end (where we need to loop around, do one-by-one)
            else:
                src_batch = []
                ids_batch = []
                for i in xrange(batch_size):
                    _id, n, r, p = data[it]
                    src_batch.append(r)
                    src_batch.append(p)
                    ids_batch.append(_id)
                    src_mols[2*i:2*i+2, :] = data_fps[h5_offset+2*it:h5_offset+2*it+2, :]
                    it = (it + 1) % data_len

            session.run(enqueue, feed_dict={_input_mol: src_mols})
            queue.put((ids_batch, src_batch))
            print('Queue size: {}'.format(queue.qsize()))
            sys.stdout.flush()

        coord.request_stop()
        f.close()

    def dummy_thread():
        return

    coord = tf.train.Coordinator()
    if interactive_mode:
        all_threads = [threading.Thread(target=dummy_thread)]
    elif test:
        all_threads = [threading.Thread(target=read_data_once, args=(opts.train_path, coord), kwargs={'frag': opts.test})]
    else:
        all_threads = [threading.Thread(target=read_data_master, args=(opts.train_path, coord))]
        print('Added read_data_master')

    [t.start() for t in all_threads]

    if not interactive_mode:
        data_len = 0
        with open(opts.train_path, 'r') as f:
            for line in f:
                data_len += 1
        print('Data length: %i' % data_len)
        if save_interval == 0: # approx once per epoch
            save_interval = np.ceil(data_len / float(batch_size))

    saver = tf.train.Saver(max_to_keep=None)
    if test or interactive_mode:
        if opts.checkpoint:
            restore_path = os.path.join(opts.save_path, 'model.%s' % opts.checkpoint)
        else:
            restore_path = tf.train.latest_checkpoint(opts.save_path)
        saver.restore(session, restore_path)
        print('Restored values from latest saved file ({})'.format(restore_path))
        test_path = '%s.prediced.%s.%s' % (restore_path, os.path.basename(opts.train_path), str(opts.test))
        summary_path = os.path.join(opts.save_path, 'model.%s.summary' % os.path.basename(opts.train_path))
    it, sum_diff, sum_gnorm, sum_diff_is_pos = 0, 0.0, 0.0, 0.0
    sum_loss = 0.0; sum_diff_is_big = 0.0
    lr = 0.001
    try:
        if interactive_mode:
            
            prompt = raw_input('enter a tag for this session: ')
            interactive_path = '%s.interactive.%s' % (restore_path, prompt.strip())
            fid = open(interactive_path, 'a')

            def get_score_from_smi(smi):
                if not smi:
                    return ('', 0.)
                src_batch = [smi]
                while len(src_batch) != (batch_size * 2): # round out last batch
                    src_batch.append('')
                src_mols = np.array(map(smi_to_fp, src_batch), dtype=np.float32)
                if sum(sum(src_mols)) == 0:
                    print('Could not get fingerprint?')
                    cur_score = [0.]
                else:
                    # Run
                    cur_score, = session.run([score], feed_dict={
                        input_mol: src_mols,
                        _lr: 0.001,
                    })
                    print('Score: {}'.format(cur_score[0]))
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
                else:
                    smi = ''
                return (smi, cur_score[0])

            while True:
                try:
                    prompt = raw_input('\nEnter SMILES (or quit): ')
                    if prompt.strip() == 'quit':
                        break
                    if str('>') in prompt: # reaction
                        reactants = prompt.strip().split('>')[0].split('.')
                        reactants_smi = []
                        reactants_score = 0.
                        for reactant in reactants:
                            (smi, cur_score) = get_score_from_smi(reactant)
                            reactants_smi.append(smi)
                            reactants_score = max(reactants_score, cur_score)
                        products = prompt.strip().split('>')[2].split('.')
                        products_smi = []
                        products_score = 0.
                        for product in products:
                            (smi, cur_score) = get_score_from_smi(product)
                            products_smi.append(smi)
                            products_score = max(products_score, cur_score)
                        smi = '{}>>{}'.format('.'.join(reactants_smi), '.'.join(products_smi))
                        fid.write('%s %s %.4f %.4f %.4f\n' % (prompt.strip(), smi, reactants_score, products_score, products_score-reactants_score))
                    else: # single or list of mols
                        reactants = prompt.strip().split('.')
                        reactants_smi = []
                        reactants_score = 0.
                        for reactant in reactants:
                            (smi, cur_score) = get_score_from_smi(reactant)
                            reactants_smi.append(smi)
                            reactants_score = max(reactants_score, cur_score)
                        fid.write('%s %s %.4f\n' % (prompt.strip(), '.'.join(reactants_smi), reactants_score))

                except KeyboardInterrupt:
                    print('Breaking out of prompt')
                    fid.close()
                    raise KeyboardInterrupt
                except Exception as e:
                    print(e)
                    fid.write('%s\n' % prompt.strip())
                    continue
        elif test:
            while queue.qsize() == 0:
                print('Letting queue fill up (5 s...)')
                time.sleep(5)

            summarystring = ''
            ctr = 0.0
            if verbose_test: 
                learned_scores = []

            sum_diff_is_pos = 0.0
            sum_diff_is_big = 0.0
            sum_diff = 0.0
            sum_gnorm = 0.0
            sum_loss = 0.0
            while True:
                try:
                    (ids_batch, src_batch) = queue.get(timeout=120)
                    if src_batch is None:
                        raise Empty
                    cur_diff, cur_score, cur_loss = session.run([diff_score, score, loss])
                    it += 1

                    for _id in ids_batch:
                        if _id > 0:
                            ctr += 1

                    sum_diff_is_pos += np.sum(cur_diff > 0)
                    sum_diff_is_big += np.sum(cur_diff > min_separation)
                    sum_diff += np.sum(cur_diff)
                    sum_loss += cur_loss
                        
                    if verbose_test:
                        for i in range(len(ids_batch)):
                            learned_scores.append(cur_score[2*i])
                            learned_scores.append(cur_score[i*2+1])

                    if it % report_interval == 0:
                        summarystring = "for %6i pairs, DiffIsPos: %.4f, DiffIs%.2f: %.4f, Loss: %.4f" % \
                            (ctr, sum_diff_is_pos / ctr, min_separation,
                            sum_diff_is_big / ctr, sum_loss / ctr)
                        print(summarystring)
                        sys.stdout.flush()

                except Empty:
                    print('End of data queue I think...have seen {} examples'.format(ctr))
                    break

            summarystring = "for %6i pairs, DiffIsPos: %.4f, DiffIs%.2f: %.4f, Loss: %.4f" % \
                (ctr, sum_diff_is_pos / ctr, min_separation, 
                 sum_diff_is_big / ctr, sum_loss / ctr)
            print(summarystring)
            sys.stdout.flush()
            fidsum = open(summary_path, 'a')
            fidsum.write('[%s-%s] %s\n' % (opts.checkpoint, opts.test, summarystring))
            fidsum.close()

            if verbose_test: 
                fid = h5py.File(test_path + '.h5', 'w')
                dset = fid.create_dataset('learned_scores', (len(learned_scores),), dtype=np.float32)
                dset[:] = np.array(learned_scores)
                fid.close()
        else:
            hist_fid = open(opts.save_path + "/model.hist", "a")

            print('Letting queue fill up (10 s)')
            time.sleep(10)
            
            while not coord.should_stop():               
                it += 1
                _, cur_diff, cur_score, pnorm, gnorm, cur_loss = session.run([backprop, diff_score, score, param_norm, grad_norm, loss], feed_dict={_lr:lr})
                (ids_batch, src_batch) = queue.get()
                
                sum_diff_is_pos += np.sum(cur_diff > 0)
                sum_diff_is_big += np.sum(cur_diff > min_separation)
                sum_diff += np.sum(cur_diff)
                sum_gnorm += gnorm
                sum_loss += cur_loss
                    
                if it % min(report_interval, save_interval) == 0:
                    logstr = "it %06i [%09i pairs seen], AvgDiff: %.2f, FracDiffPos: %.3f, FracDiff%.2f: %.3f, PNorm: %.2f, GNorm: %.2f, Loss: %.4f" % \
                        (it, it*batch_size*2, sum_diff / (report_interval * batch_size), 
                            sum_diff_is_pos / (report_interval * batch_size),
                            min_separation, sum_diff_is_big / (report_interval * batch_size),
                            pnorm, sum_gnorm / report_interval,
                            sum_loss / report_interval) 
                    hist_fid.write(logstr + "\n")
                    print(logstr)
                    sys.stdout.flush()
                    sum_diff, sum_gnorm, sum_perfrank = 0.0, 0.0, 0.0
                    sum_loss, sum_diff_is_pos, sum_diff_is_big = 0.0, 0.0, 0.0

                    print('Ex: {:.2f}>>{:.2f} -> diff = {:.2f}'.format(
                        cur_score[0], cur_score[1], cur_diff[0]))
                    print('Ex: ID{} === {}>>{}'.format(
                        ids_batch[0], src_batch[0], src_batch[1]))

                    sys.stdout.flush()

                if it % save_interval == 0:
                    lr *= 0.9
                    saver.save(session, opts.save_path + "/model.ckpt", global_step=it)
                    print "Model Saved! Decaying learning rate"

                if it >= max(min_iterations, max_save * save_interval):
                    coord.request_stop()

    except Exception as e:
        print e
        coord.request_stop(e)
    finally:
        if not test and not interactive_mode: 
            saver.save(session, opts.save_path + "/model.final")
            hist_fid.close()
        coord.request_stop()
        coord.join(all_threads)
        try:
            [p.join() for p in processes]
        except Exception:
            pass
