#!/bin/sh

mkdir gallery
for v in {0..3}
do
	python3 scripts/view.py 0 $v
done
echo "DONE!"
