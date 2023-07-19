#!/bin/sh


NVIDIA_SMI_ARGS="-lms 1 --format=csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used"
AUTOCORR_ARGS="-I 30 -r"
OUTPUT_FOLDER="output/g32-l10"

if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
    echo "Folder created: $OUTPUT_FOLDER"
fi

echo "Running for 10.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f "${OUTPUT_FOLDER}/10-utilization.csv" & pid2=$!
./main -p 10 -i ../input/10.csv  -o "${OUTPUT_FOLDER}/10.csv" $AUTOCORR_ARGS > "${OUTPUT_FOLDER}/10" & pid1=$!

wait $pid1
sleep 1
kill $pid2

sleep 1

echo "Running for 100.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f "${OUTPUT_FOLDER}/100-utilization.csv" & pid4=$!
./main -p 10 -i ../input/100.csv  -o "${OUTPUT_FOLDER}/100.csv" $AUTOCORR_ARGS > "${OUTPUT_FOLDER}/100" & pid3=$!

wait $pid3
sleep 1
kill $pid4

sleep 1

echo "Running for 1000.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f "${OUTPUT_FOLDER}/1000-utilization.csv" & pid6=$!
./main -p 100 -i ../input/1000.csv  -o "${OUTPUT_FOLDER}/1000.csv" $AUTOCORR_ARGS > "${OUTPUT_FOLDER}/1000" & pid5=$!

wait $pid5
sleep 1
kill $pid6

sleep 1

echo "Running for 10000.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f "${OUTPUT_FOLDER}/10000-utilization.csv" & pid8=$!
./main -p 1000 -i ../input/10000.csv  -o "${OUTPUT_FOLDER}/10000" $AUTOCORR_ARGS > "${OUTPUT_FOLDER}/10000" & pid7=$!

wait $pid7
sleep 1
kill $pid8

sleep 1

echo "Running for 30000.csv"

nvidia-smi $NVIDIA_SMI_ARGS -f "${OUTPUT_FOLDER}/30000-utilization.csv" & pid10=$!
./main -p 1000 -i ../input/30000.csv  -o "${OUTPUT_FOLDER}/30000" $AUTOCORR_ARGS > "${OUTPUT_FOLDER}/30000" & pid9=$!

wait $pid9
sleep 1
kill $pid10

echo "Completed!"