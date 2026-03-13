#!/bin/bash

set -u  # fail on undefined variables

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_training () {
    script=$1
    logfile=$2

    log "Starting $script"

    ./$script > "$logfile" 2>&1
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log "Finished $script successfully"
    else
        log "WARNING: $script failed with exit code $exit_code"
    fi
}

log "Training queue started"

run_training run_aff2nff_cycleloss_2_noprepro.sh run_output_03013_aff2nff_raw_cl2_noprep.txt
#run_training run_hel2nff_cycleloss_2.sh run_output_03012_hel2nff_raw_cl2.txt
run_training run_aff2nff_cycleloss_5_noprepro.sh run_output_03013_aff2nff_raw_cl5_noprep.txt
#run_training run_hel2nff_cycleloss_5.sh run_output_03012_hel2nff_raw_cl5.txt
run_training run_aff2nff_cycleloss_10_noprepro.sh run_output_03013_aff2nff_raw_cl10_noprep.txt

log "Training queue finished"