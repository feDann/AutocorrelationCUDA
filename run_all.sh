#!/bin/sh


NVIDIA_SMI_ARGS="-lms 1 --format=csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used"
AUTOCORR_ARGS="-I 50"

echo "Running for 10.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f output/10-utilization.csv & pid2=$!
./main -p 10 -i ../input/10.csv  -o output/10 $AUTOCORR_ARGS & pid1=$!

wait $pid1
sleep 1
kill $pid2

sleep 1

echo "Running for 100.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f output/100-utilization.csv & pid4=$!
./main -p 10 -i ../input/100.csv  -o output/100 $AUTOCORR_ARGS & pid3=$!

wait $pid3
sleep 1
kill $pid4

sleep 1

echo "Running for 1000.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f output/1000-utilization.csv & pid6=$!
./main -p 100 -i ../input/1000.csv  -o output/1000 $AUTOCORR_ARGS & pid5=$!

wait $pid5
sleep 1
kill $pid6

sleep 1

echo "Running for 10000.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f output/10000-utilization.csv & pid8=$!
./main -p 1000 -i ../input/10000.csv  -o output/10000 $AUTOCORR_ARGS & pid7=$!

wait $pid7
sleep 1
kill $pid8

sleep 1

echo "Running for 30000.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f output/30000-utilization.csv & pid10=$!
./main -p 1000 -i ../input/30000.csv  -o output/30000 $AUTOCORR_ARGS & pid9=$!

wait $pid9
sleep 1
kill $pid10

echo "Completed!"