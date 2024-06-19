import pandas as pd
import os
from tqdm import tqdm
from IPython.display import clear_output
from src.pipeline import FinancePipeline
from multimodal.financial.src.utils import download_raw_texts_from_urls, save_text_to, load_text_from
import logging

if __name__ == "__main__":
    logging.basicConfig(
        filename='log.txt',  # Set the filename for the log file
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Set the format for log messages
        datefmt='%Y-%m-%d %H:%M:%S'  # Set the format for the date in log messages
    )
    logger = logging.getLogger()

    directory_path = '/data/kai/forecasting/raw_urls'
    file_names = os.listdir(directory_path)

    ticker = "aapl"
    df = pd.read_csv(directory_path + f'/{ticker}_text.csv')[::-1]

    data_dir = "/data/kai/forecasting/summary"
    pipe = FinancePipeline(data_dir)
    pipe.set_prompts_for_ticker(ticker)


    for idx, date_str in enumerate(df["timestamp"].unique()):
        if os.path.exists(f"{data_dir}/{ticker}_{date_str}_final_summary.txt"):
            # logging.info(f"Skipping {ticker}_{date_str}")
            continue

        logger.info(f"Running {idx} / {len(df["timestamp"].unique())} urls for {date_str} for {ticker}")
        current_urls = df[df["timestamp"] == date_str]["url"]

        raw_path = f"{data_dir}/{ticker}_{date_str}_raw.txt"
        if os.path.exists(raw_path):
            raw_texts = load_text_from(raw_path)
            raw_texts = [r for r in raw_texts if r != "<SEP>"]
        else:
            raw_texts = download_raw_texts_from_urls(current_urls)
            save_text_to(raw_texts, raw_path)

        summary_path = f"{data_dir}/{ticker}_{date_str}_summaries.txt"
        if os.path.exists(summary_path):
            summaries = load_text_from(summary_path)
            summaries = [s for s in summaries if s !="" and s != "<SEP>"]
        else:
            summaries = pipe.get_summaries(raw_texts)
            save_text_to(summaries, summary_path)
        
        combined_summary = pipe.combine_summaries(summaries)

        if len(combined_summary) == 1:
            final_summary = pipe.run_llama([pipe.filter_prompt, combined_summary[0]])

        if len(combined_summary) > 1:
            logger.info("Combinig summarys one more time...")
            combined_summary_again = pipe.combine_summaries(combined_summary)
            final_summary = pipe.run_llama([pipe.filter_prompt, combined_summary_again[0]])

        save_text_to(final_summary, f"{data_dir}/{ticker}_{date_str}_final_summary.txt")

        final_sentiment = pipe.run_llama(["Provide a sentiment for the following stock news:", final_summary])
        logger.info(final_sentiment)
