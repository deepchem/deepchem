#!/bin/bash
cd v2015
for x in */
do
  cd $x
  babel -isdf ${x%/}_ligand.sdf -opdb ${x%/}_ligand.pdb
  cd ../
done
cd ../
