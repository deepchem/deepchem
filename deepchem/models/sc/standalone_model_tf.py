'''
This is the code for a standalone importable SCScorer model. It relies on tensorflow 
and simply reinitializes from a save file. 

One method dumps the trainable variables as numpy arrays, which then enables the 
standalone_model_numpy version of this class.
'''

import tensorflow as tf
from utils.nn import linearND
import math, sys, random, os
import numpy as np
import time
import rdkit.Chem as Chem 
import rdkit.Chem.AllChem as AllChem

import os 
project_root = os.path.dirname(os.path.dirname(__file__))

score_scale = 5.0
min_separation = 0.25

FP_len = 1024
FP_rad = 2
batch_size = 2

class SCScorer():
    def __init__(self):
        self.session = tf.Session()

    def build(self, depth=5, hidden_size=300, score_scale=score_scale, FP_len=FP_len, FP_rad=FP_rad):
        self.FP_len = FP_len; self.FP_rad = FP_rad
        self.input_mol = tf.placeholder(tf.float32, [batch_size*2, FP_len])
        self.mol_hiddens = tf.nn.relu(linearND(self.input_mol, hidden_size, scope="encoder0"))
        for d in xrange(1, depth):
            self.mol_hiddens = tf.nn.relu(linearND(self.mol_hiddens, hidden_size, scope="encoder%i"%d))

        self.score_sum = linearND(self.mol_hiddens, 1, scope="score_sum")
        self.score_sum = tf.squeeze(self.score_sum)
        self.score = 1.0 + (score_scale - 1.0) * tf.nn.sigmoid(self.score_sum)

        tf.global_variables_initializer().run(session=self.session)
        size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
        n = sum(size_func(v) for v in tf.trainable_variables())
        print "Model size: %dK" % (n/1000,)

        self.coord = tf.train.Coordinator()
        return self

    def restore(self, model_path, checkpoint='final'):
        self.saver = tf.train.Saver(max_to_keep=None)
        restore_path = os.path.join(model_path, 'model.%s' % checkpoint)
        self.saver.restore(self.session, restore_path)
        print('Restored values from latest saved file ({})'.format(restore_path))
        
        if 'uint8' in model_path or 'counts' in model_path:
            def mol_to_fp(self, mol):
                if mol is None:
                    return np.array((self.FP_len,), dtype=np.uint8)
                fp = AllChem.GetMorganFingerprint(mol, self.FP_rad, useChirality=True) # uitnsparsevect
                fp_folded = np.zeros((self.FP_len,), dtype=np.uint8)
                for k, v in fp.GetNonzeroElements().iteritems():
                    fp_folded[k % self.FP_len] += v 
                return np.array(fp_folded)
        else:
            def mol_to_fp(self, mol):
                if mol is None:
                    return np.zeros((self.FP_len,), dtype=np.float32)
                return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.FP_rad, nBits=self.FP_len, 
                    useChirality=True), dtype=np.bool)
        self.mol_to_fp = mol_to_fp
        return self 

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def get_score_from_smi(self, smi='', v=False):
        if not smi:
            return ('', 0.)
        src_batch = [smi]
        while len(src_batch) != (batch_size * 2): # round out last batch
            src_batch.append('')
        src_mols = np.array(map(self.smi_to_fp, src_batch), dtype=np.float32)
        if sum(sum(src_mols)) == 0:
            if v: print('Could not get fingerprint?')
            cur_score = [0.]
        else:
            # Run
            cur_score, = self.session.run([self.score], feed_dict={
                self.input_mol: src_mols,
            })
            if v: print('Score: {}'.format(cur_score[0]))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ''
        return (smi, cur_score[0])

    def dump_to_numpy_arrays(self, dump_path):
        import cPickle as pickle
        with open(dump_path, 'wb') as fid:
            pickle.dump([v.eval(session=self.session) for v in tf.trainable_variables()], fid, -1)


if __name__ == '__main__':
    model = SCScorer()
    model.build()
    model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024bool'), 'ckpt-10654')
    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        (smi, sco) = model.get_score_from_smi(smi)
        print('%.4f <--- %s' % (sco, smi))
    model.dump_to_numpy_arrays(os.path.join(project_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.pickle'))

    # model = SCScorer()
    # model.build(FP_len=2048)
    # model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_2048bool'), 'ckpt-10654')
    # smis = ['CCCOCCC', 'CCCNc1ccccc1']
    # for smi in smis:
    #     (smi, sco) = model.get_score_from_smi(smi)
    #     print('%.4f <--- %s' % (sco, smi))
    # model.dump_to_numpy_arrays(os.path.join(project_root, 'models', 'full_reaxys_model_2048bool', 'model.ckpt-10654.as_numpy.pickle'))

    # model = SCScorer()
    # model.build()
    # model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024uint8'), 'ckpt-10654')
    # smis = ['CCCOCCC', 'CCCNc1ccccc1']
    # for smi in smis:
    #     (smi, sco) = model.get_score_from_smi(smi)
    #     print('%.4f <--- %s' % (sco, smi))
    # model.dump_to_numpy_arrays(os.path.join(project_root, 'models', 'full_reaxys_model_1024uint8', 'model.ckpt-10654.as_numpy.pickle'))