#!/bin/bash

for i in {0..6}; do
    echo $i;
    qsub engine.sh &
done
