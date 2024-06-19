SUMMARY_PROMPT = f"""
You are a helpful assistant that filters and summarizes stock news specifically for the company with ticker symbol {ticker}. 

Your task is to:
1. Filter out irrelevant information.
2. Provide a concise summary that includes key numbers, growth trends, and the overall market outlook.
3. Mention major stock movements, significant economic indicators, and any notable company-specific news.
4. Avoid making up any information.

If there is no relevant information, the website is blocked, or there is an error message, return nothing.
"""

IGNORE_PROMPT = f"""
Return "<NONE>" if there is an error in the summarization or if the information is irrelevant. Otherwise, return "<TRUE>".
"""

COMBINE_PROMPT = f"""
Combine the summaries into one by following the guidelines:

1. Provide a concise summary that includes key numbers, growth trends, and the overall market outlook.
2. Mention major stock movements, significant economic indicators, and any notable company-specific news.
"""

FINAL_PROMPT = f"""
Given the following stock summaries related to the company with ticker symbol {ticker},

1. Combine them into one comprehensive summary
2. Provide key numbers, growth trends, and the overall market outlook.
3. Mention major stock movements, significant economic indicators, and any notable company-specific news.
4. Avoid making up any information.
"""