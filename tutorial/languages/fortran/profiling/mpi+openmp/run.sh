#!/bin/bash

./build.sh

./gen-strong-scaling.sh
echo "Done generating strong scaling script"

./strong-scaling.sh
echo "Done running strong scaling"

./gen-weak-scaling.sh
echo "Done generating weak scaling script"

./weak-scaling.sh
echo "Done running weak scaling"