#!/bin/bash

for i in {0..15}; do
    echo $i;
    qsub engine_single.sh &
done
