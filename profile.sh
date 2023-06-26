#!/bin/bash


echo "Profiling with input: 10"
nvprof -f -s --analysis-metrics --log-file ./profiler/10_summary  -o ./profiler/10.nvvp ./main -p 10 -i ../input/10.csv -o ./profiler/t 

echo "Profiling with input: 100"
nvprof -f -s --analysis-metrics --log-file ./profiler/100_summary  -o ./profiler/100.nvvp ./main -p 10 -i ../input/100.csv -o ./profiler/t 

echo "Profiling with input: 1000"
nvprof -f -s --analysis-metrics --log-file ./profiler/1000_summary  -o ./profiler/1000.nvvp ./main -p 100 -i ../input/1000.csv -o ./profiler/t 

echo "Profiling with input: 10000"
nvprof -f -s --analysis-metrics --log-file ./profiler/10000_summary  -o ./profiler/10000.nvvp ./main -p 1000 -i ../input/10000.csv -o ./profiler/t 

echo "Profiling with input: 30000"
nvprof -f -s --analysis-metrics --log-file ./profiler/30000_summary  -o ./profiler/30000.nvvp ./main -p 1000 -i ../input/30000.csv -o ./profiler/t 