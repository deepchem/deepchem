import os
import re



def replace_epochs(fname):
    lines = [x.strip('\n') for x in open(fname).readlines()]
    new_lines = []
    for line in lines:
        if line.find('"NB_EPOCH = ') != -1:
            new_lines.append('"NB_EPOCH = 1\\n",')
            continue
        new_lines.append(line)
    with open(fname, 'w') as fout:
        for line in new_lines:
            fout.write("%s\n" % line)

def main():
    notebooks = os.listdir('examples/notebooks')
    notebooks = list(filter(lambda x: x.endswith('.ipynb'), notebooks))
    for notebook in notebooks:
        notebook = os.path.join('examples/notebooks', notebook)
        replace_epochs(notebook)


if __name__ == "__main__":
    main()
