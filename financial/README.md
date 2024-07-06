
Go to multimodal/financial directory and run in following order with appropriate arguments.
Modify .env's API_KEY and BASE_URL for vLLM inference.

1. python -m scripts.download_raw_urls
2. python -m scripts.summarize_documents --ticker="AAPL" --config_path="config/summary_v0.2.yaml"
3. python -m scripts.document_to_summary
4. python -m scripts.format_summary