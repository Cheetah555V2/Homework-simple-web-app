import random
import time
import os
import hashlib
from typing import Any

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
