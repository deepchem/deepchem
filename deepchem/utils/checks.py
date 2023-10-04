import json
import os

with open(os.path.join(os.path.dirname(__file__), "assets", "periodic_table.json")) as read_file:
    m = json.load(read_file)

print(m["atom_masses"])

