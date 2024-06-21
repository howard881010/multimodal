#!/bin/bash

run_script() {
  python main.py --ticker "$1" &
}

# tickers=("value1" "value2" "value3")
tickers=($(ls $directory))


for ticker in "${tickers[@]}"; 
do
  run_script "$ticker"
done

wait

echo "All scripts have finished."
