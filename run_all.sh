#!/bin/bash

./main -p 10 -i ../input/10.csv  -o outputs/10 >> outputs/10.csv
./main -p 10 -i ../input/100.csv  -o outputs/100 >> outputs/100.csv
./main -p 100 -i ../input/1000.csv  -o outputs/1000 >> outputs/1000.csv
./main -p 1000 -i ../input/10000.csv  -o outputs/10000 >> outputs/10000.csv
./main -p 1000 -i ../input/30000.csv  -o outputs/30000 >> outputs/30000.csv