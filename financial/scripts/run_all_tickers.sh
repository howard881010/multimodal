#!/bin/bash

max_parallel=3

run_script() {
  python summary.py --ticker "$1" --config_path="/data/kai/forecasting/multimodal/financial/config/summary_v0.2.yaml" &
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
