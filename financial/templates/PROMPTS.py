# system is high-level instruction
# user is queries or prompt
# assistant is model's response

class Prompts:
    def __init__(self, ticker: str = None):
        # VLLM prompts
        self.SUMMARY_PROMPT = \
            f"""You are a helpful assistant that filters and summarizes stock news specifically for the company with ticker symbol {ticker}. 

Your task is to:
1. Filter out irrelevant information.
2. Provide a concise summary that includes key numbers, growth trends, and the overall market outlook.
3. Mention major stock movements, significant economic indicators, and any notable company-specific news.
4. Avoid making up any information.
5. Provide a concise summary without using introductory phrases like 'Here is a summary of ___' or similar. Focus directly on the key points.

If there is no relevant information, the website is blocked, or there is an error message, return nothing.
"""

        self.IGNORE_PROMPT = \
            f"""Return "<NONE>" if there is an error in the summarization or if the information is irrelevant. Otherwise, return "<TRUE>".
"""

        self.COMBINE_PROMPT = \
            f"""Combine the summaries into one by following the guidelines:

1. Provide a concise summary that includes key numbers, growth trends, and the overall market outlook.
2. Mention major stock movements, significant economic indicators, and any notable company-specific news.
3. Provide a concise summary without using introductory phrases like 'Here is a summary of ___' or similar. Focus directly on the key points.
"""

        self.FINAL_PROMPT = \
            f"""Given the following stock summaries related to the company with ticker symbol {ticker},

1. Combine them into one comprehensive summary
2. Provide key numbers, growth trends, and the overall market outlook.
3. Mention major stock movements, significant economic indicators, and any notable company-specific news.
4. Avoid making up any information.
5. Provide a concise summary without using introductory phrases like 'Here is a summary of ___' or similar. Focus directly on the key points.
"""
        # JSON Guided Prompts
        self.JSON_SUMMARY_PROMPT = \
            f"""You are a helpful assistant for converting raw text of a stock news website into relevant text information specifically for the company with ticker symbol {ticker}.

1. Include key_numbers, growth_trends, overall_market_outlook, major_stock_movements, significant_economic_indicators, notable_company_specific_news, and a final summary.
2. Provide as much information as you can by always adding relevant units or details.
3. Avoid making up any information.
"""
        self.COMBINE_JSON_PROMPT = \
            f"""For the list of stock news of {ticker} in JSON format, combine them into one json format.

1. Combine the list of key_numbers, growth_trends, overall_market_outlook, major_stock_movements, significant_economic_indicators, notable_company_specific_news, and a final summary.
2. Reorder the text as needed.
2. Remove repetitive details and always add relevant units or details
3. Avoid making up any information and repeating information.
4. Provide a concise summary without using introductory phrases like 'Here is a summary of ___' or similar. Focus directly on the key points.
"""


class ForecstBaselinePrompts:
    def __init__(self, window: int = 5):

        self.SYSTEM_PROMPT = \
            f"""You are an expert financial forecaster. Given each day's price and summary, predict the next {window} days prices. Provide {window} numerical values only, seperated by commas and no other text.
"""


blocked_words = [
    "thestreet.comPlease enable JS and disable any ad blocker",
    "wsj.comPlease enable JS and disable any ad blocker",
    "Access Denied Access Denied You don't have permission to access",
    "Access to this page has been denied",
    "Sorry! Temporarily Unavailable Sorry, this page is temporarily unavailable for technical reasons."
]
