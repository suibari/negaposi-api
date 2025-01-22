from flask import Flask, request, jsonify
from janome.tokenizer import Tokenizer
import nltk
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
    # 全体の結果を格納する変数
    wakati_results = []  # 2次元配列
    average_sentiments = []  # 1次元配列
    noun_stats_map = {}  # 名詞の集計を保持する辞書（再集計用）

    # 各テキストのリストに対して処理
    for text in texts:
        # 個々のテキストを analyze_sentiment に渡す
        result = analyze_sentiment(text)

        # 分かち書き結果を保存
        wakati_results.append(result["wakati"])

        # 感情スコアを保存
        average_sentiments.append(result["sentiment"])

        # 名詞集計を再集計
        for noun_data in result["nouns_count"]:
            noun = noun_data["noun"]
            count = noun_data["count"]
            sentiment_sum = noun_data["sentiment_sum"]

            if noun not in noun_stats_map:
                noun_stats_map[noun] = {"count": 0, "sentiment_sum": 0}

            noun_stats_map[noun]["count"] += count
            noun_stats_map[noun]["sentiment_sum"] += sentiment_sum

    # 名詞集計結果をリスト形式に変換
    nouns_counts = [
        {"noun": noun, "count": stats["count"], "sentiment_sum": stats["sentiment_sum"]}
        for noun, stats in noun_stats_map.items()
    ]

    return {
        "wakati": wakati_results,  # 2次元配列
        "average_sentiments": average_sentiments,  # 1次元配列
        "nouns_counts": nouns_counts  # 1次元配列
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

def sanitize_text(text, max_length=10000):
    """
    入力文字列をクリーンアップする関数。
    制御文字・非ASCII文字を除去し、日本語、アルファベット、数字のみを許可。

    Args:
        text (str): 入力テキスト。
        max_length (int): 許容する最大文字列長。

    Returns:
        str: クリーンアップされた文字列。
    """
    # 空文字チェック
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        # 制御文字を除去
        text = re.sub(r"[\x00-\x1F\x7F]", "", text)

        # アルファベットと半角英数と記号と改行とタブを排除
        text = re.sub(r'[a-zA-Z0-9¥"¥.¥,¥@]+', '', text)
        text = re.sub(r'[!"“#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}~]', '', text)
        text = re.sub(r'[\n|\r|\t]', '', text)

        # 日本語以外の文字を排除(韓国語とか中国語とかヘブライ語とか)
        jp_chartype_tokenizer = nltk.RegexpTokenizer(u'([ぁ-んー]+|[ァ-ンー]+|[\u4e00-\u9FFF]+|[ぁ-んァ-ンー\u4e00-\u9FFF]+)')
        text = "".join(jp_chartype_tokenizer.tokenize(text))

        # テキストの長さを制限
        if len(text) > max_length:
            text = text[:max_length]
    except Exception as e:
        # エラーが発生した場合は空文字を返す
        print(f"Error sanitizing text: {e}")
        return ""

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