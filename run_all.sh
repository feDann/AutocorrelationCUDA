#!/bin/sh

set -e

TIMEPOINTS=30000

NVIDIA_SMI_ARGS="-lms 1 --format=csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used"
AUTOCORR_ARGS="-I 30 -r -p ${TIMEPOINTS}"
OUTPUT_FOLDER="output"
INPUT_FOLDER="../../thesis/input"


LEVELS_CONFIG="8 10 16"
GROUP_SIZE_CONFIG="8 16 32"


echo "Running benchmarks for $TIMEPOINTS timepoints"

for level in $LEVELS_CONFIG; do
    for group in $GROUP_SIZE_CONFIG; do

        CONFIG_OUTPUT="$OUTPUT_FOLDER/g$group-l$level"

        if [ ! -d "$CONFIG_OUTPUT" ]; then
            mkdir -p "$CONFIG_OUTPUT"
            echo "Folder created: $CONFIG_OUTPUT"
        fi

        echo "Running current config: -l $level -g $group for $TIMEPOINTS timepoints"

        nvidia-smi $NVIDIA_SMI_ARGS -f "${CONFIG_OUTPUT}/${TIMEPOINTS}-utilization.csv" & pid2=$!
        ./bin/main $AUTOCORR_ARGS -i "${INPUT_FOLDER}/${TIMEPOINTS}.csv"  -l "${level}" -g "${group}" -o "${CONFIG_OUTPUT}/${TIMEPOINTS}.csv" > "${CONFIG_OUTPUT}/${TIMEPOINTS}" & pid1=$!


        wait $pid1
        sleep 1
        kill $pid2

        sleep 1

    done
done