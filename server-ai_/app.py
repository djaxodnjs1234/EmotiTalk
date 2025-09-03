from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
import numpy as np
import torch.nn.functional as F

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU를 사용합니다.")
else:
    device = torch.device("cpu")
    print("CPU를 사용합니다.")

# 1. 학습된 모델과 토크나이저 로드
MODEL_DIR = "./model"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
except Exception as e:
    print(f"모델 또는 토크나이저를 로드하는 데 실패했습니다. 'model' 디렉토리가 존재하는지 확인해주세요. 오류: {e}")
    exit()

# 2. 감성 레이블을 수동으로 정의
id_to_label = {
    0: '공포',
    1: '놀람',
    2: '분노',
    3: '슬픔',
    4: '중립',
    5: '행복',
    6: '혐오'
}

# 3. 예측 함수 정의
def predict_sentiment_with_confidence(text, top_n=2):
    model.eval()

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().flatten()

    sorted_indices = np.argsort(probabilities)[::-1]

    top_results = []
    for i in range(top_n):
        index = sorted_indices[i]
        label = id_to_label[index]
        probability = probabilities[index]
        top_results.append((label, probability))
    return top_results

# 감정 강도 차이에 따른 분류 함수
def classify_emotion_expression(diff: float) -> int:
    AGGRESSIVE_THRESHOLD = 0.65
    PASSIVE_THRESHOLD = 0.1
    if diff >= AGGRESSIVE_THRESHOLD:
        return 0        # 적극적 감정 표출
    elif diff <= PASSIVE_THRESHOLD:
        return 2        # 감정 표출 없음
    else:
        return 1        # 소극적 감정 표출

# 감정+이모티콘 매칭 함수
def matching_emotion(emotion1: str, result: int) -> str:
        if emotion1 == '공포':  # 공포
            if result == 0:
                return '😱'
            elif result == 1:
                return '😨'
            else: return None

        elif emotion1 == '놀람':  # 놀람
            if result == 0:
                return '😲'
            elif result == 1:
                return '😯'
            else: return None

        elif emotion1 == '분노':  # 분노
            if result == 0:
                return '😡'
            elif result == 1:
                return '😤'
            else: return None

        elif emotion1 == '슬픔':  # 슬픔
            if result == 0:
                return '😢'
            elif result == 1:
                return '😞'
            else: return None

        elif emotion1 == '중립':  # 중립
            return None

        elif emotion1 == '행복':  # 행복
            if result == 0:
                return '😄'
            elif result == 1:
                return '🙂'
            else: return None

        elif emotion1 == '혐오':  # 혐오
            if result == 0:
                return '🤢'
            elif result == 1:
                return '=😒'
            else: return None
        else:
            return "이런 감정은 없는댑쇼"

# Flask 앱 설정
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/", methods=["POST"])
def analyze_emotion():
    data = request.get_json()
    sentence = data.get("sentence", "")
    if not sentence:
        return jsonify({"error": "No sentence provided."}), 400

    top_sentiments = predict_sentiment_with_confidence(sentence, top_n=2)
    first_sentiment, first_prob = top_sentiments[0]
    second_sentiment, second_prob = top_sentiments[1]

    prob_diff = first_prob - second_prob
    expression_class = classify_emotion_expression(prob_diff)
    emoji_desc = matching_emotion(first_sentiment, expression_class)
    result_text = f"{sentence} {emoji_desc}"
    emoji = emoji_desc[-1] if emoji_desc else ''

    # 콘솔 출력도 OK
    print("----------------")
    print(f"요청받은 문장: {sentence}")
    print(f"주감정: {first_sentiment} ({first_prob:.4f}) / 보조감정: {second_sentiment} ({second_prob:.4f}) / 차이: {prob_diff*100:.2f}%")
    print(f"최종 결과: {emoji if emoji is not None else '중립'}")
    print("----------------")

    # 여기서 emoji와 텍스트 '따로' 보내줌
    return jsonify({"text": sentence, "emoji": emoji, "result": result_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)