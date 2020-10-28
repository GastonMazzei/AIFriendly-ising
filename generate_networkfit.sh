#!/bin/sh

mkdir gallery
echo "INFO: About to train a neural network. Classification of the Magnetization for the threshold 0.5 with a bi-layer with 3 neurons per layer is the default behaviour. \n Running 'python3 scripts/network_class.py NEURONS EPOCHS THRESHOLD' is an alternative for custom conditions\ne.g. 'python3 scripts/network_class.py 5 120 0.7'"
for i in {1,2,3,4}
do
  python3 scripts/network_class.py 20 50 $i 0.5
done
echo "SUCCESS!"
