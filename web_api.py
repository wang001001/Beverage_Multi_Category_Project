from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BertProductAPI:
    """BERT商品分类API类"""

    def __init__(self, model_path='model'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.load_model()

    def load_model(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.model_path}")
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(device)
        self.model.eval()  # 设置为评估模式
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.label_encoder = joblib.load(f'{self.model_path}/label_encoder.pkl')

    def predict_single(self, text):
        """预测单个文本"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            pred = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probs).item()

            pred_label = self.label_encoder.inverse_transform([pred])[0]

        return pred_label, confidence


# 初始化API实例
api = BertProductAPI()


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()

        if 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400

        text = data['text']

        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({'error': 'Invalid text provided'}), 400

        # 进行预测
        prediction, confidence = api.predict_single(text)

        response = {
            'success': True,
            'prediction': prediction,
            'confidence': round(float(confidence), 4),
            'input_text': text
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测接口"""
    try:
        data = request.get_json()

        if 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400

        texts = data['texts']

        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'Invalid texts provided, expected a non-empty list'}), 400

        results = []
        for text in texts:
            if not isinstance(text, str) or len(text.strip()) == 0:
                results.append({
                    'text': text,
                    'error': 'Invalid text'
                })
                continue

            prediction, confidence = api.predict_single(text)
            results.append({
                'text': text,
                'prediction': prediction,
                'confidence': round(float(confidence), 4)
            })

        response = {
            'success': True,
            'results': results
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': api.model is not None,
        'device': str(device)
    })


@app.route('/', methods=['GET'])
def home():
    """主页"""
    return jsonify({
        'message': 'BERT Product Classification API',
        'endpoints': {
            'predict': '/predict (POST)',
            'batch_predict': '/batch_predict (POST)',
            'health': '/health (GET)'
        }
    })


if __name__ == '__main__':
    print("启动BERT商品分类API服务...")
    print("设备:", device)
    print("模型路径:", api.model_path)
    app.run(host='0.0.0.0', port=8010, debug=False)