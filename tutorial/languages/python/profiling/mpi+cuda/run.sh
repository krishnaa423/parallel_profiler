#!/bin/bash

./gen-weak-scaling.sh
echo "Done generating weak scaling script"

./weak-scaling.sh
echo "Done running weak scaling"