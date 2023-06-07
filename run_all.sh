#!/bin/bash

./main -p 10 -i ../input/10.csv  -o output/10 > output/10.csv
./main -p 10 -i ../input/100.csv  -o output/100 > output/100.csv
./main -p 100 -i ../input/1000.csv  -o output/1000 > output/1000.csv
./main -p 1000 -i ../input/10000.csv  -o output/10000 > output/10000.csv
./main -p 1000 -i ../input/30000.csv  -o output/30000 > output/30000.csv