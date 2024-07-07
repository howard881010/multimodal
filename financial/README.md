# Launch vLLM in Nautilus
To launch vLLM in nautilus, modify .env's API_KEY and BASE_URL

```bash
InternalServerError: <html><body><h1>503 Service Unavailable</h1>
No server is available to handle this request.
</body></html>
```
*Currently port issue with vLLM. Can't connect to port 8000.

1. kubectl apply -f pvc.yaml: Create persistant volume claim for storage:
2. kubectl apply -f llama_deployment.yaml: Deploy vLLM image
3. kubectl apply -f ingress.yaml: Configure network and ports
4.  kubectl exec -it -f llama_deployment.yaml -- /bin/bash: launch interactive shell to debug

# Scripts
Go to multimodal/financial directory and run in following order with appropriate arguments.
Modify .env's API_KEY and BASE_URL for vLLM inference.

1. python -m scripts.download_raw_urls
2. python -m scripts.summarize_documents --ticker="AAPL" --config_path="config/summary_v0.2.yaml"
3. python -m scripts.document_to_summary
4. python -m scripts.format_summary