from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.cosmoshub.tech/v1"
)

try:
    response = client.chat.completions.create(
        model="claude-sonnet-4.5",  # atau model yang kamu pakai
        messages=[
            {"role": "user", "content": "Halo"}
        ]
    )

    print(response.choices[0].message.content)

except Exception as e:
    print(e)