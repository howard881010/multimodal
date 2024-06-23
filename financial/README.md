
* running on roselab1 2*A100
```bash
export CUDA_VISIBLE_DEVICES='0,1'
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size=2 --disable-log-requests
```

*modify data_path as needed

1) run download_raw_urls.py to download raw text from all URLS
2) run ./run_all_tickers.sh to run main.py in parallel to summarize all
3) run format_to_csv.py to get formatted csv for each company
