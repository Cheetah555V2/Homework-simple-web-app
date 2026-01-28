from mistralai import Mistral

import dotenv
import os

dotenv.load_dotenv()

# Initialize client
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

res = client.models.list()
print(res)


# response = client.chat.complete(
#     model="mistral-large-latest",
#     messages=[
#         {
#             "role": "user",
#             "content": "What is the capital of France?"
#         }
#     ]
# )

# print(response.choices[0].message.content)