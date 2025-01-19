#!/bin/bash

# 引数が指定されているか確認
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 'text to analyze'"
  exit 1
fi

# 渡されたテキストを変数に格納
TEXT=$1

# APIエンドポイントのURL
API_URL="http://127.0.0.1:5000/analyze"

# POSTリクエストを送信
response=$(curl -s -X POST -H "Content-Type: application/json" -d "{\"text\": \"$TEXT\"}" $API_URL)

# 結果を表示
echo "Response from API:"
echo $response
