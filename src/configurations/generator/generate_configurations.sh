#!/bin/bash 

for i in $(seq 3 15);
do
    python3 conf_generator.py --grid-size $i
done
