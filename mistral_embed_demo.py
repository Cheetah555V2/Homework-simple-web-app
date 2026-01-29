from mistralai import Mistral
import dotenv
import os
import numpy as np

dotenv.load_dotenv()

# Initialize client
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

input_texts = [
    "I'm thinking about the purpose of life. It's very interesting and can lead to deep insights and meaningfull conversations with like-minded people.",
    "What is bothering me now is how to get out of the class faster and go for a lunch",
    "I have and interesting idea of tech-based IT project, like to have more free time to experiment and make it work faster!",
    "I'm leading in my class leaderboard for clash royale mobile game, plan to train more and get a place in the competition team",
    "I have found nice manga but read all the chapter available for now - and the next should be published next month... sad...",
    "For the long time seen no interesting anime - plan the check what is available now and may be spend a day with it!",
    "Tired sitting in the class and with the laptop... want to do some sports, maybe badminton or bicicle!"
]

res = client.embeddings.create(model="mistral-embed", inputs=input_texts)
base_embeddings = [res.data[i].embedding for i in range(len(input_texts))]
print(base_embeddings[0])

# Helper function for cosine similarity
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Compare base embeddings
sims = []
for i in range(len(base_embeddings)):
    for j in range(i+1, len(base_embeddings)):
        similarity = cosine_similarity(base_embeddings[i], base_embeddings[j])
        sims.append((similarity, i, j))

sims.sort(reverse=True)
print("Most similar pairs:")
for sim, i, j in sims[:5]:
    print(f"{sim:.4f}: [{i}] vs [{j}]")
    print(f"  {input_texts[i][:60]}...")
    print(f"  {input_texts[j][:60]}...")
    print()

# Find match for new sentence
new_sentence = "Looking for a company to play badminton"
res = client.embeddings.create(model="mistral-embed", inputs=[new_sentence])
new_emb = res.data[0].embedding

similarities = []
for i in range(len(base_embeddings)):
    sim = cosine_similarity(new_emb, base_embeddings[i])
    similarities.append((sim, i))

similarities.sort(reverse=True)
print(f"\nMatches for: '{new_sentence}'")
for sim, i in similarities[:5]:
    print(f"{sim:.4f}: [{i}] {input_texts[i][:80]}...")