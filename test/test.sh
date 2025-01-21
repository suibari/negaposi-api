#!/bin/bash

# 引数が指定されているか確認
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 'text1' 'text2' ... 'textN'"
  echo "Provide multiple texts as arguments for analysis."
  exit 1
fi

# 複数の引数をスペースで区切ってリスト化
TEXTS=("$@")

# JSON形式の1次元配列を作成
TEXT_ARRAY=$(printf '%s\n' "${TEXTS[@]}" | jq -R -s -c 'split("\n")[:-1]')

# APIエンドポイントのURL
API_URL="http://127.0.0.1:5000/analyze"

# POSTリクエストを送信
response=$(curl -s -X POST -H "Content-Type: application/json" -d "{\"texts\": $TEXT_ARRAY}" $API_URL)

# 結果を表示
echo "Response from API:"
echo $response
