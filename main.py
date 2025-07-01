from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from janome.tokenizer import Tokenizer
import nltk
import pandas as pd
import os
import re
from typing import List

app = FastAPI()

# 日本語辞書の読み込み
df = pd.read_csv('./dict/pn.csv.m3.120408.trim', header=None, delimiter='\t')
word_dict = dict(zip(df[0], df[1]))

# 英語辞書の読み込み（pn_en.dic形式: word:POS:score）
en_dict_path = './dict/pn_en.dic'
en_word_dict = {}
with open(en_dict_path, encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(':')
        if len(parts) == 3:
            word, _, score = parts
            try:
                en_word_dict[word.lower()] = float(score)
            except ValueError:
                continue

# 形態素解析器の初期化
tokenizer = Tokenizer()

# データ送信形式を定義する
class TextsRequest(BaseModel):
    texts: List[str]

# メイン処理
def analyze_texts(texts: List[str]):
    wakati_results = []
    average_sentiments = []
    noun_stats_map = {}
    nouns_results = []

    for text in texts:
        if contains_japanese(text):
            sanitized = sanitize_text(text)
            result = analyze_sentiment(sanitized)
        else:
            result = analyze_sentiment_en(text)

        wakati_results.append(result["wakati"])
        average_sentiments.append(result["sentiment"])
        nouns_results.append(result["nouns"])

        for noun_data in result["nouns_count"]:
            noun = noun_data["noun"]
            count = noun_data["count"]
            sentiment_sum = noun_data["sentiment_sum"]
            if noun not in noun_stats_map:
                noun_stats_map[noun] = {"count": 0, "sentiment_sum": 0}
            noun_stats_map[noun]["count"] += count
            noun_stats_map[noun]["sentiment_sum"] += sentiment_sum
    
    nouns_counts = [
        {"noun": noun, "count": stats["count"], "sentiment_sum": stats["sentiment_sum"]}
        for noun, stats in noun_stats_map.items()
    ]

    return {
        "wakati": wakati_results,
        "average_sentiments": average_sentiments,
        "nouns": nouns_results,
        "nouns_counts": nouns_counts
    }

# 日本語を含むか判定
def contains_japanese(text: str) -> bool:
    return bool(re.search(r'[\u3040-\u30FF\u4E00-\u9FFF]', text))

# テキスト感情解析(日本語)
def analyze_sentiment(text: str):
    tokens = tokenizer.tokenize(text)  # トークンをジェネレーターで取得

    if not tokens:
        return {
            "sentiment": 0,  
            "wakati": [],  
            "nouns_count": [],
            "nouns": [],
        }

    detailed_tokens = []
    noun_stats_map = {}  
    wakati = []  
    total_score = 0
    token_count = 0

    for token in tokens:
        surface = token.surface
        wakati.append(surface)  

        token_info = {
            "token": surface,
            "part_of_speech": token.part_of_speech.split(',')[0],
            "sentiment": 0  
        }

        if '名詞' in token.part_of_speech:
            if surface not in noun_stats_map:
                noun_stats_map[surface] = {"count": 0, "sentiment_sum": 0}

            noun_stats_map[surface]["count"] += 1

            if surface in word_dict:
                score = word_dict[surface]
                if score == 'p':  
                    token_info["sentiment"] = 1
                    total_score += 1
                    noun_stats_map[surface]["sentiment_sum"] += 1
                elif score == 'n':  
                    token_info["sentiment"] = -1
                    total_score -= 1
                    noun_stats_map[surface]["sentiment_sum"] -= 1

            token_count += 1

        detailed_tokens.append(token_info)

    average_sentiment = total_score / token_count if token_count > 0 else 0

    noun_stats = [
        {"noun": noun, "count": stats["count"], "sentiment_sum": stats["sentiment_sum"]}
        for noun, stats in noun_stats_map.items()
    ]
    nouns = list(noun_stats_map.keys())

    return {
        "sentiment": average_sentiment,
        "wakati": wakati,
        "nouns": nouns,
        "nouns_count": noun_stats
    }

# テキスト感情解析(英語)
def analyze_sentiment_en(text: str):
    print(text)
    words = re.findall(r'\b\w+\b', text.lower())  # 英単語のみ抽出
    wakati = []
    total_score = 0
    token_count = 0
    noun_stats_map = {}

    for word in words:
        if len(word) < 2:
            continue
        sentiment = en_word_dict.get(word, 0.0)
        if word not in noun_stats_map:
            noun_stats_map[word] = {"count": 0, "sentiment_sum": 0}
        noun_stats_map[word]["count"] += 1
        noun_stats_map[word]["sentiment_sum"] += sentiment
        if sentiment != 0.0:
            total_score += sentiment
            token_count += 1
        wakati.append(word)

    average_sentiment = total_score / token_count if token_count > 0 else 0

    noun_stats = [
        {"noun": noun, "count": stats["count"], "sentiment_sum": stats["sentiment_sum"]}
        for noun, stats in noun_stats_map.items()
    ]
    nouns = list(noun_stats_map.keys())

    return {
        "sentiment": average_sentiment,
        "wakati": wakati,
        "nouns": nouns,
        "nouns_count": noun_stats
    }

# テキストクリーンアップ
def sanitize_text(text: str, max_length: int = 10000) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        text = re.sub(r"[\x00-\x1F\x7F]", "", text)
        text = re.sub(r'[a-zA-Z0-9¥"¥.¥,¥@]+', '', text)
        text = re.sub(r'[!"“#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}~]', '', text)
        text = re.sub(r'[\n|\r|\t]', '', text)

        jp_chartype_tokenizer = nltk.RegexpTokenizer(u'([ぁ-んー]+|[ァ-ンー]+|[\u4e00-\u9FFF]+|[ぁ-んァ-ンー\u4e00-\u9FFF]+)')
        text = "".join(jp_chartype_tokenizer.tokenize(text))

        if len(text) > max_length:
            text = text[:max_length]
    except Exception as e:
        print(f"Error sanitizing text: {e}")
        return ""

    return text

# APIエンドポイント定義
@app.post("/analyze")
async def analyze_text(request: TextsRequest):
    texts = request.texts
    result = analyze_texts(texts)
    return result

@app.get("/ping")
async def ping():
    return {"message": "pong"}

# Uvicornで起動時の設定
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
