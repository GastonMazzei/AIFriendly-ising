#!/bin/sh

echo "INFO: About to train a neural network. Classification of the Magnetization for the threshold 0.5 with a bi-layer with 3 neurons per layer is the default behaviour. \n Running 'python3 scripts/network_class.py NEURONS EPOCHS THRESHOLD' is an alternative for custom conditions\ne.g. 'python3 scripts/network_class.py 5 120 0.7'"
python3 scripts/network_class.py 3 70 0.5
echo "SUCCESS!"
