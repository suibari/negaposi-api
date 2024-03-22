from flask import Flask, request, jsonify
from janome.tokenizer import Tokenizer
import pandas as pd

app = Flask(__name__)

# 評価極性辞書を読み込む
df = pd.read_csv('./dict/pn.csv.m3.120408.trim', header=None, delimiter='\t')
word_dict = dict(zip(df[0], df[1]))

# 形態素解析器の初期化
tokenizer = Tokenizer()

def analyze_sentiment(text):
    tokens = tokenizer.tokenize(text)
    total_score = 0
    token_count = 0
    for token in tokens:
        surface = token.surface
        print(surface)
        if surface in word_dict:
            score = word_dict[surface]
            if score == 'p':
                total_score += 1
            elif score == 'n':
                total_score -= 1
            # 'e' の場合は何もしない
            token_count += 1
    if token_count == 0:
        return 0  # トークンがない場合は0を返す
    return total_score / token_count

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if data is None or 'text' not in data:
        return jsonify({'error': 'Invalid input. Please provide a JSON object with "text" field.'}), 400
    
    text = data['text']
    sentiment = analyze_sentiment(text)
    result = {'text': text, 'sentiment': sentiment}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)