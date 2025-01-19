from flask import Flask, request, jsonify
from janome.tokenizer import Tokenizer
import pandas as pd
import os
import re

app = Flask(__name__)

# 評価極性辞書を読み込む
df = pd.read_csv('./dict/pn.csv.m3.120408.trim', header=None, delimiter='\t')
word_dict = dict(zip(df[0], df[1]))

# 形態素解析器の初期化
tokenizer = Tokenizer()

def analyze_texts(texts):
    # 全体の結果を格納するリスト
    wakati_results = []
    average_sentiments = []
    nouns_counts_results = []

    # 各テキストのリストに対して処理
    for text_group in texts:
        wakati_group = []  # 現在のグループの分かち書き結果
        sentiment_group = []  # 現在のグループの感情スコア
        nouns_count_group = []  # 現在のグループの名詞集計

        for text in text_group:
            # 個々のテキストを analyze_sentiment に渡す
            result = analyze_sentiment(text)

            # 結果を各リストに追加
            wakati_group.append(result["wakati"])
            sentiment_group.append(result["sentiment"])
            nouns_count_group.append(result["nouns_count"])

        # グループごとの結果を全体リストに追加
        wakati_results.append(wakati_group)
        average_sentiments.append(sentiment_group)
        nouns_counts_results.append(nouns_count_group)

    return {
        "wakati": wakati_results,
        "average_sentiments": average_sentiments,
        "nouns_counts": nouns_counts_results
    }

def analyze_sentiment(text):
    tokens = tokenizer.tokenize(sanitize_text(text))  # トークンをジェネレーターで取得
    detailed_tokens = []
    noun_stats_map = {}  # 名詞の集計を保持する辞書
    wakati = []  # 分かち書き結果
    total_score = 0
    token_count = 0

    for token in tokens:
        surface = token.surface
        wakati.append(surface)  # 分かち書きのために追加
        
        token_info = {
            "token": surface,
            "part_of_speech": token.part_of_speech.split(',')[0],  # 最初の品詞だけを取得
            "sentiment": 0  # デフォルト値
        }

        # 名詞の場合のみ処理を行う
        if '名詞' in token.part_of_speech:
            if surface not in noun_stats_map:
                noun_stats_map[surface] = {"count": 0, "sentiment_sum": 0}

            noun_stats_map[surface]["count"] += 1

            # 感情スコアが辞書にあれば取得して加算
            if surface in word_dict:
                score = word_dict[surface]
                if score == 'p':  # ポジティブスコア
                    token_info["sentiment"] = 1
                    total_score += 1
                    noun_stats_map[surface]["sentiment_sum"] += 1
                elif score == 'n':  # ネガティブスコア
                    token_info["sentiment"] = -1
                    total_score -= 1
                    noun_stats_map[surface]["sentiment_sum"] -= 1
                # 'e' の場合は sentiment をそのまま 0 にする
            
            token_count += 1

        detailed_tokens.append(token_info)

    # 平均スコアの計算
    average_sentiment = total_score / token_count if token_count > 0 else 0

    # 名詞の集計結果を辞書からリスト形式に変換
    noun_stats = [
        {"noun": noun, "count": stats["count"], "sentiment_sum": stats["sentiment_sum"]}
        for noun, stats in noun_stats_map.items()
    ]

    return {
        "sentiment": average_sentiment,  # テキスト全体の感情値
        "wakati": wakati,  # 分かち書き結果
        "nouns_count": noun_stats  # 名詞の集計結果
    }

def sanitize_text(text):
    # 空文字チェック
    if not isinstance(text, str) or not text.strip():
        return ""

    # 制御文字を除去
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    # 非ASCII文字や適切でないUnicode文字のフィルタ
    text = re.sub(r"[^\w\sぁ-んァ-ヶ一-龠々ー]", "", text)

    return text

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if data is None or 'texts' not in data:
        return jsonify({'error': 'Invalid input. Please provide a JSON object with "texts" field.'}), 400
    
    texts = data['texts']
    result = analyze_texts(texts)
    return jsonify(result)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200

if __name__ == '__main__':
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))

    app.run(host=host, port=port)