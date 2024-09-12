from huggingface_hub import snapshot_download

sql_lora_path = snapshot_download(repo_id="Howard881010/climate-1day")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="unsloth/Meta-Llama-3.1-8B-Instruct", enable_lora=True, device="cuda:0", gpu_memory_utilization=1.0)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=4096,
    stop=["[/assistant]"],
    seed=42
)

prompts = [
     """[user] Given the weather information of the first 1 day, predict the weather information of the next 1 day. Output the result strictly in the following JSON format and no additional text: { "day_2_date": "YYYY-MM-DD", "day_2_weather_forecast": "Weather description" }
      { "day_1_date": "2014-01-01", "day_1_weather_forecast": "Arctic air is expected to invade the upper Midwest, Great Lakes, Ohio Valley, and Northeast, bringing some of the coldest temperatures in recent memory. Significant cold outbreaks on January 3-4 and 5-8 could result in temperatures 20-30 degrees Fahrenheit below normal. Lows in New England may reach -10s to -20s, and areas in eastern Montana, Minnesota, and Wisconsin may experience lows at or below -20F. Colder locations could drop to -30s or -40s Fahrenheit. Moderate upslope snow is anticipated across east-facing slopes from Montana to Northeast Wyoming. The coldest day is projected around January 8, with temperatures below freezing in a large section of the northern United States, potentially marking the coldest air masses since 1993/1994.\n\nPrecipitation is expected from a frontal zone moving across the Midwest and East on Sunday and Monday, followed by lake effect snows from late Monday into Wednesday. Radiational cooling could contribute to extremely low temperatures in the Ohio Valley and Mid-Atlantic states. The Gold Coast of Florida will remain slightly above normal, with temperatures struggling to drop below 60\u00b0F. \n\nThe western region will experience milder weather, with parts of the Great Basin and Southwest being 10+ degrees Fahrenheit above normal. Light to moderate precipitation is likely across the Pacific Northwest into the Northern and Central Rockies Tuesday and Wednesday, with potential for a Santa Ana event in Southern California. Temperatures across the Northern Continental Divide should rebound after mid-level heights increase on Wednesday." } [/user] [assistant]""",
     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
]

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path)
)

for output in outputs:
    print(output.outputs[0].text)
    print("\n\n")