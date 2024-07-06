# system is high-level instruction
# user is queries or prompt
# assistant is model's response
class Prompts:
    def __init__(self, ticker: str = None):
        self.ticker = ticker

    @property
    def summary_prompt(self):
        return (
            f"You are a helpful assistant that filters and summarizes stock news specifically for the company with ticker symbol {self.ticker}. \n\n"
            "Your task is to:\n"
            "1. Filter out irrelevant information.\n"
            "2. Provide a concise summary that includes key numbers, growth trends, and the overall market outlook.\n"
            "3. Mention major stock movements, significant economic indicators, and any notable company-specific news.\n"
            "4. Avoid making up any information.\n"
            "5. Provide a concise summary without using introductory phrases like 'Here is a summary of ___' or similar. Focus directly on the key points.\n\n"
            "If there is no relevant information, the website is blocked, or there is an error message, return nothing."
        )

    @property
    def ignore_prompt(self):
        return 'Return "<NONE>" if there is an error in the summarization or if the information is irrelevant. Otherwise, return "<TRUE>".'

    @property
    def combine_prompt(self):
        return (
            "Combine the summaries into one by following the guidelines:\n\n"
            "1. Provide a concise summary that includes key numbers, growth trends, and the overall market outlook.\n"
            "2. Mention major stock movements, significant economic indicators, and any notable company-specific news.\n"
            "3. Provide a concise summary without using introductory phrases like 'Here is a summary of ___' or similar. Focus directly on the key points."
        )

    @property
    def final_prompt(self):
        return (
            f"Given the following stock summaries related to the company with ticker symbol {self.ticker},\n\n"
            "1. Combine them into one comprehensive summary\n"
            "2. Provide key numbers, growth trends, and the overall market outlook.\n"
            "3. Mention major stock movements, significant economic indicators, and any notable company-specific news.\n"
            "4. Avoid making up any information.\n"
            "5. Provide a concise summary without using introductory phrases like 'Here is a summary of ___' or similar. Focus directly on the key points."
        )

    @property
    def json_summary_prompt(self):
        return (
            f"You are a helpful assistant for converting raw text of a stock news website into relevant text information specifically for the company with ticker symbol {self.ticker}.\n\n"
            "1. Include key_numbers, growth_trends, overall_market_outlook, major_stock_movements, significant_economic_indicators, notable_company_specific_news, and a final summary.\n"
            "2. Provide as much information as you can by always adding relevant units or details.\n"
            "3. Avoid making up any information."
        )

    @property
    def combine_json_prompt(self):
        return (
            f"For the list of stock news of {self.ticker} in JSON format, combine them into one JSON format.\n\n"
            "1. Combine the list of key_numbers, growth_trends, overall_market_outlook, major_stock_movements, significant_economic_indicators, notable_company_specific_news, and a final summary.\n"
            "2. Reorder the text as needed.\n"
            "3. Remove repetitive details and always add relevant units or details.\n"
            "4. Avoid making up any information and repeating information.\n"
            "5. Provide a concise summary without using introductory phrases like 'Here is a summary of ___' or similar. Focus directly on the key points."
        )


class ForecastBaselinePrompts:
    def __init__(self, window: int = 5):
        self.window = window

    @property
    def system_prompt(self):
        return (
            f"You are an expert financial forecaster. Given each day's price and summary, predict the next {self.window} days prices. Provide {self.window} numerical values only, separated by commas and no other text."
        )
