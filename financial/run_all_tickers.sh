#!/bin/bash

max_parallel=3

run_script() {
  python summarize_chunk.py --ticker "$1" &
}

directory="/data/kai/forecasting/data/raw_v0.2_filtered"
tickers=($(ls $directory))

for ticker_with_extension in "${tickers[@]}";
do
  ticker="${ticker_with_extension%.csv}"

  while [ $(jobs -r | wc -l) -ge $max_parallel ]; do
    sleep 1
  done

  run_script "$ticker"
done
wait
