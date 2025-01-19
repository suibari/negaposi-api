#!/bin/bash

# 引数が指定されているか確認
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 'text1' 'text2' ... 'textN'"
  echo "Provide multiple texts as arguments to form a 2D array for analysis."
  exit 1
fi

# 複数の引数をスペースで区切ってグループ化
TEXTS=("$@")

# 2次元配列を作成（例として2グループに分割）
GROUP1=("${TEXTS[@]:0:$((${#TEXTS[@]} / 2))}")
GROUP2=("${TEXTS[@]:$((${#TEXTS[@]} / 2))}")

# JSON形式の二次元配列を作成
TEXT_ARRAY=$(jq -n --argjson group1 "$(printf '%s\n' "${GROUP1[@]}" | jq -R -s -c 'split("\n")[:-1]')" \
                     --argjson group2 "$(printf '%s\n' "${GROUP2[@]}" | jq -R -s -c 'split("\n")[:-1]')" \
                     '[$group1, $group2]')

# APIエンドポイントのURL
API_URL="http://127.0.0.1:5000/analyze"

# POSTリクエストを送信
response=$(curl -s -X POST -H "Content-Type: application/json" -d "{\"texts\": $TEXT_ARRAY}" $API_URL)

# 結果を表示
echo "Response from API:"
echo $response
