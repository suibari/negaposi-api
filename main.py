from flask import Flask, request, jsonify
from janome.tokenizer import Tokenizer
import pandas as pd
import os

app = Flask(__name__)

# 評価極性辞書を読み込む
df = pd.read_csv('./dict/pn.csv.m3.120408.trim', header=None, delimiter='\t')
word_dict = dict(zip(df[0], df[1]))

# 形態素解析器の初期化
tokenizer = Tokenizer()

def analyze_sentiment(text):
    tokens = tokenizer.tokenize(text)
    wakati = []
    total_score = 0
    token_count = 0
    for token in tokens:
        wakati.append(token.surface)
        if '名詞' in token.part_of_speech:
            surface = token.surface
            if surface in word_dict:
                score = word_dict[surface]
                if score == 'p':
                    total_score += 1
                elif score == 'n':
                    total_score -= 1
                # 'e' の場合は何もしない
                token_count += 1
    if token_count == 0:
        return {'sentiment': 0, 'tokens': wakati}
    return {'sentiment': total_score / token_count, 'tokens': wakati}

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if data is None or 'text' not in data:
        return jsonify({'error': 'Invalid input. Please provide a JSON object with "text" field.'}), 400
    
    text = data['text']
    result = analyze_sentiment(text)
    return jsonify(result)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200

if __name__ == '__main__':
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))

    app.run(host=host, port=port)