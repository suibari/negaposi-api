from flask import Flask, request, jsonify
import oseti

app = Flask(__name__)

# OSETIの初期化
analyzer = oseti.Analyzer()

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    # POSTリクエストからJSONデータを取得
    data = request.json
    if data is None or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    # 感情分析
    scores = analyzer.analyze(data['text'])

    # 結果を整形
    response = {
        'sentiment_scores': scores
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)