#!/bin/bash

max_parallel=3

run_script() {
  echo "running ticker '$1'"
  python main.py --ticker "$1" &
}

directory="/data/kai/forecasting/summary"
tickers=($(ls $directory))
# assume following file structure:
  # summary
    # ticker_1
    # ticker_2
    # ...

for ticker in "${tickers[@]}"; 
do
  # Check if there are 2 or more running jobs
  while [ $(jobs -r | wc -l) -ge $max_parallel ]; do
    sleep 1
  done
  run_script "$ticker"
done

wait

echo "All scripts have finished."
