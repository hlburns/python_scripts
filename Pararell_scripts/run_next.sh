#!/bin/bash

while [ -d /proc/367936 ]; do sleep 1; done && ipython parallel_test.py > p_2.txt &

