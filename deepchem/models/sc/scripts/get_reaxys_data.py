import os
from pymongo import MongoClient
import rdkit.Chem as Chem

'''
Get examples from Reaxys where...
(a) we can parse the reactants and products
(b) there is a single product (product salts == multiple products)
(c) there is at least one instance that is explicitly single step

This is meant to work with data hosted in a MongoDB

While we can't include the actual data, this shows our preprocessing pipeline. The saved file 
consists of the reaction smiles string, the maximum number of atoms in the reactants or products, 
and the document ID for traceability.
'''

limit = 10 # small for demonstration

client = MongoClient('mongodb://username:password@host.address/admin', 27017)
reaction_db = client['database_name']['reactions']
instance_db = client['database_name']['instances']

project_root = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(project_root, 'data', 'reaxys_limit%i.txt' % limit), 'w') as f:
	i = 0
	for rx_doc in reaction_db.find({'RXN_SMILES': {'$exists': True}}, ['_id', 'RXN_SMILES', 'RX_NVAR']).sort('_id', 1):
		try:
			r, p = rx_doc['RXN_SMILES'].split('>>')
			if (not r) or (not p) or ('.' in p):
				continue
			r_mol = Chem.MolFromSmiles(str(r))
			p_mol = Chem.MolFromSmiles(str(p))
			if (not r_mol) or (not p_mol): 
				continue
			rxd_id_list = ['%i-%i' % (rx_doc['_id'], j) for j in range(1, int(rx_doc['RX_NVAR']) + 1)]
			single_step = False
			for rxd_doc in instance_db.find({'_id': {'$in': rxd_id_list}}, ['RXD_STP']):
				if rxd_doc['RXD_STP'] == ['1']:
					single_step = True 
					break
			if not single_step:
				continue
			[a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
			[a.ClearProp('molAtomMapNumber') for a in p_mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
			n = max(r_mol.GetNumAtoms(), p_mol.GetNumAtoms())
			f.write('%s>>%s %i %i\n' % (Chem.MolToSmiles(r_mol,True), Chem.MolToSmiles(p_mol,True), n, rx_doc['_id']))
			i += 1
			if i % 1000 == 0:
				print('Wrote %i' % i)
			if i >= limit:
				break
		except Exception as e:
			print(e)
