try:
    from neuralfingerprint.mol_graph import graph_from_smiles_tuple, degrees
    from neuralfingerprint import mol_graph
    from data_parser import parse_graph
except:
    pass

import numpy as np

class Molecule(object):
    def __init__(self, atoms, adj_mat, type_adj, deg_list, bonds):
        self.atoms = atoms
        self.adj_mat = adj_mat
        self.type_adj = type_adj
        self.deg_list = deg_list
        self.bonds = bonds    
        
    def get_atoms(self):
        return self.atoms
    
    def get_adj_mat(self):
        return self.adj_mat

    def get_type_adj(self):
        return self.type_adj
    
    def get_deg_list(self):
        return self.deg_list
    
    def get_bonds(self):
        return self.bonds

class SmilesDataManager():
    def __init__(self, raw_smiles, targets, bond_decimals):
        self.raw_smiles = raw_smiles
        self.targets = targets

        # Run parser
        self.run(bond_decimals)

    def run(self, bond_decimals):
        self.parse_data(bond_decimals)

    def get_N_molecules(self):
        return len(self.raw_smiles)
        
    def parse_data(self, bond_decimals):
        self.molecules = []

        k = 0
        while k < self.get_N_molecules():
            smile = self.raw_smiles[k]
            target = self.targets[k]

            # Convert smile to graph
            #print(target, smile)
            graph = mol_graph.graph_from_smiles(smile)

            # Get features
            try:
                atoms, adj_mat, type_adj, deg_list, bonds = parse_graph(graph, bond_decimals)
                self.molecules.append(Molecule(atoms, adj_mat, type_adj, deg_list, bonds))
            except:
                # Remove the bad example
                self.raw_smiles = np.delete(self.raw_smiles, k)
                self.targets = np.delete(self.targets, k)
                print("molecule with smile " + smile + " failed to compile")
            k += 1
