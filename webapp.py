import random
import time
import os
import hashlib
from typing import Any
import json
import numpy as np
import logging
from datetime import datetime

from flask import Flask, render_template, request

from calcs import plot_expression

# Docs and examples for Flask: https://flask.palletsprojects.com/en/stable/
app = Flask(__name__)  # To run, use flask --app webapp run --debug

# add Mistral client support
import dotenv
dotenv.load_dotenv()
from mistralai import Mistral
import re

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
client = None
if MISTRAL_API_KEY:
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception:
        client = None

# Convert free-form user request into a python expression using Mistral (fallback to heuristic)
def convert_request_to_expr(user_request: str) -> str:
    if not user_request:
        return ''

    # try using Mistral if client available
    if client is not None:
        try:
            system_msg = {
                "role": "system",
                "content": (
                    "You are a converter that turns a user's free-form plotting request into a single-line Python expression in terms of 'x'. "
                    "Use numpy (np) where appropriate (e.g. np.sin(x)). Reply with the expression only, no explanations, no surrounding backticks."
                )
            }
            user_msg = {"role": "user", "content": user_request}
            # cast messages to Any to satisfy the SDK's typing in strict linters
            messages: Any = [system_msg, user_msg]
            resp = client.chat.complete(model="mistral-large-latest", messages=messages, temperature=0)

            # defensively extract the textual content from the response
            raw = getattr(resp.choices[0].message, 'content', None)
            content = ''
            if isinstance(raw, str):
                content = raw
            elif isinstance(raw, list):
                parts = []
                for chunk in raw:
                    if isinstance(chunk, str):
                        parts.append(chunk)
                    elif isinstance(chunk, dict) and 'text' in chunk:
                        parts.append(chunk['text'])
                    elif hasattr(chunk, 'text'):
                        parts.append(getattr(chunk, 'text'))
                content = ''.join([p for p in parts if p])
            elif raw is None:
                content = ''
            else:
                try:
                    content = str(raw)
                except Exception:
                    content = ''

            # strip code fences and whitespace safely
            content = re.sub(r"^```(?:python)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()
            # if multi-line, take first non-empty line
            for line in content.splitlines():
                line = line.strip()
                if line:
                    # remove common prefixes like 'Expression:'
                    line = re.sub(r"^[A-Za-z ]*:", "", line).strip()
                    return line
        except Exception:
            # fall back to heuristic below
            pass

    # simple local heuristic fallback
    s = user_request.lower()
    if 'sine' in s or 'sin(' in s or 'sin ' in s:
        return 'np.sin(x)'
    if 'cos' in s or 'cosine' in s:
        return 'np.cos(x)'
    if 'tan' in s or 'tangent' in s:
        return 'np.tan(x)'
    if 'square' in s or 'x**2' in s or 'squared' in s:
        return 'x**2'
    if 'cube' in s or 'x**3' in s or 'cubic' in s:
        return 'x**3'
    if 'exponential' in s or 'exp(' in s or 'e^' in s:
        return 'np.exp(x)'
    if 'log' in s:
        return 'np.log(x)'
    # try to extract inline code-like expression between backticks
    m = re.search(r"`([^`]+)`", user_request)
    if m:
        return m.group(1)
    return 'x'

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, 'friend_msgs.json')

def load_messages():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_messages(msgs):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)

def cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

@app.route("/")  # http://127.0.0.1:5000/
def main_page():
    plot_file = ''
    version = ''
    user_input = request.args.get('func_expr', '').strip()

    min_interval = request.args.get('min_interval', type=float)
    max_interval = request.args.get('max_interval', type=float)

    color = request.args.get('color', 'blue', type=str)

    print(color, color[0:2])

    # set sensible defaults if not provided
    if min_interval is None:
        min_interval = -2.0
    if max_interval is None:
        max_interval = 2.0

    expr_str = ''
    if user_input != '':
        expr_str = convert_request_to_expr(user_input)

        os.makedirs('static', exist_ok=True) # make sure that dir static exist

        unhash_text = expr_str + '&' + str(min_interval) + '&' + str(max_interval) + '&' + str(color)
        h = hashlib.sha256(unhash_text.encode('utf-8')).hexdigest()[:12]
        plot_file = f"plot_{h}.png"
        plot_path = os.path.join('static', plot_file)

        try:
            plot_expression(expr_str, min_interval, max_interval, color, plot_path)
            version = str(int(time.time() * 1000))  # cache-buster for the browser
        except Exception:
            plot_file = ''
            version = ''

    return render_template('plot_func.html',
                           func_expr=user_input,
                           expr_str=expr_str,
                           plot_file=plot_file,
                           version=version,
                           min_interval=min_interval,
                           max_interval=max_interval,
                           color=color)

@app.route("/test")  # http://127.0.0.1:5000/test
def test_route():
    x = random.randint(0, 10)

    return render_template('main_page.html', lucky_num=x)

# replace simple friend_finder route with a full POST/GET handler
@app.route("/friend-finder", methods=["GET", "POST"]) 
def friend_finder():
    if request.method == 'GET':
        return render_template('friend_finder.html', recommendations=None, top3=None)

    nickname = request.form.get('nickname', '').strip() or 'Anonymous'
    message = request.form.get('message', '').strip()

    if not message:
        return render_template('friend_finder.html', error='Please enter a message', recommendations=None, top3=None)

    # compute embedding using Mistral if available
    emb = None
    if client is not None:
        try:
            res = client.embeddings.create(model='mistral-embed', inputs=[message])
            emb = res.data[0].embedding
        except Exception:
            logging.exception('Embedding API call failed, falling back to local embedding')
            emb = None

    if emb is None:
        # deterministic fallback embedding: simple char histogram vector
        vec = np.zeros(256, dtype=float)
        for ch in message:
            vec[ord(ch) % 256] += 1.0
        norm = np.linalg.norm(vec) + 1e-12
        emb = (vec / norm).tolist()

    msgs = load_messages()
    sims = []
    for m in msgs:
        sim = cosine_sim(emb, m.get('embedding', []))
        sims.append((sim, m))

    sims.sort(key=lambda x: x[0], reverse=True)
    top3 = sims[:3]

    logging.info(f"New message by {nickname}: {message}")
    logging.info("Top-3 candidates: " + ", ".join([f"{m['nickname']} ({sim:.4f})" for sim, m in top3]))

    recommendations = []

    # Use LLM to filter the top-3 candidates if available
    if client is not None and top3:
        try:
            list_text = []
            for i, (sim, m) in enumerate(top3, start=1):
                list_text.append(f"{i}) nickname: {m['nickname']}. message: {m['message']}. similarity: {sim:.4f}")

            prompt = (
                "You are an assistant that selects which of the listed messages are good friend matches for the new user's message. "
                "Reply with a comma-separated list of numbers (1..3) corresponding to items that are clearly relevant and share interests, or reply 'none'.\n\n" + "\n".join(list_text)
            )
            system_msg = {"role": "system", "content": "You are a converter that selects relevant friend matches."}
            user_msg = {"role": "user", "content": prompt}
            messages: Any = [system_msg, user_msg]
            resp = client.chat.complete(model="mistral-large-latest", messages=messages, temperature=0)

            raw = getattr(resp.choices[0].message, 'content', None)
            text = ''
            if isinstance(raw, str):
                text = raw
            elif isinstance(raw, list):
                parts = []
                for chunk in raw:
                    if isinstance(chunk, str):
                        parts.append(chunk)
                    elif isinstance(chunk, dict) and 'text' in chunk:
                        parts.append(chunk['text'])
                    elif hasattr(chunk, 'text'):
                        parts.append(getattr(chunk, 'text'))
                text = ''.join(parts)
            elif raw is None:
                text = ''
            else:
                try:
                    text = str(raw)
                except Exception:
                    text = ''

            nums = re.findall(r"\d+", text)
            chosen = set(int(n) for n in nums if 1 <= int(n) <= len(top3))
            for idx in sorted(chosen):
                sim, m = top3[idx - 1]
                recommendations.append({"nickname": m['nickname'], "message": m['message'], "sim": float(sim)})
        except Exception:
            logging.exception('LLM filtering failed')

    else:
        # fallback rule: recommend candidates with sim > 0.55
        for sim, m in top3:
            if sim > 0.55:
                recommendations.append({"nickname": m['nickname'], "message": m['message'], "sim": float(sim)})

    # append and save the new message
    msgs.append({"nickname": nickname, "message": message, "embedding": emb, "ts": datetime.utcnow().isoformat()})
    save_messages(msgs)

    return render_template('friend_finder.html', recommendations=recommendations, top3=[{"nickname": m['nickname'], "message": m['message'], "sim": float(sim)} for sim, m in top3])