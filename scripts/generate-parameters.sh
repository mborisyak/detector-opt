#!/bin/bash

batch=128
energies="0.25 0.5 0.75 1.0"
designs=128

echo 'Energies' $energies
echo 'Desgins' ${designs}

seed=9999

for e in $energies; do
  i=0
  while [ $i -lt $designs ]; do
    low="$i"
    high=$(( low + batch - 1 ))
    echo $e $low $high $seed
    (( seed += batch ))
    (( i += batch ))
  done
done > parameters.txt
