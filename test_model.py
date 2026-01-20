import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import joblib
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class ProductDataset(Dataset):
    """商品数据集类"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertProductTester:
    """BERT商品分类测试器"""

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
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.label_encoder = joblib.load(f'{self.model_path}/label_encoder.pkl')

    def load_test_data(self, test_path):
        """加载测试数据"""
        test_data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                test_data.append(data)

        return pd.DataFrame(test_data)

    def create_data_loader(self, texts, labels, batch_size=16, shuffle=False):
        """创建数据加载器"""
        dataset = ProductDataset(texts, labels, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return accuracy, f1, predictions, true_labels

    def predict_single(self, text):
        """预测单个文本"""
        self.model.eval()

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

    def predict_batch(self, texts):
        """批量预测"""
        predictions = []
        confidences = []

        for text in tqdm(texts, desc="Predicting"):
            pred_label, confidence = self.predict_single(text)
            predictions.append(pred_label)
            confidences.append(confidence)

        return predictions, confidences


def main():
    print("初始化BERT商品分类测试器...")

    # 创建测试器实例
    tester = BertProductTester()

    print("正在加载测试数据...")
    test_df = tester.load_test_data('data/test.jsonl')

    print(f"测试数据大小: {len(test_df)}")
    print(f"类别数量: {test_df['category'].nunique()}")

    # 准备测试数据
    test_texts = test_df['product_name'].tolist()
    test_labels_encoded = tester.label_encoder.transform(test_df['category'].tolist())

    # 创建测试数据加载器
    test_loader = tester.create_data_loader(test_texts, test_labels_encoded, batch_size=16)

    # 评估模型
    print("开始评估模型...")
    test_acc, test_f1, test_predictions, test_true = tester.evaluate(test_loader)

    # 解码预测结果
    test_pred_labels = tester.label_encoder.inverse_transform(test_predictions)
    test_true_labels = tester.label_encoder.inverse_transform(test_true)

    print(f"\n测试集最终结果:")
    print(f"准确率: {test_acc:.4f}")
    print(f"F1分数: {test_f1:.4f}")

    print(f"\n详细分类报告:")
    print(classification_report(test_true_labels, test_pred_labels))

    # 展示预测示例
    print(f"\n=== 预测示例 ===")
    sample_indices = np.random.choice(len(test_df), size=min(10, len(test_df)), replace=False)
    sample_texts = test_df.iloc[sample_indices]['product_name'].tolist()
    sample_actual = test_df.iloc[sample_indices]['category'].tolist()

    for i, (text, actual) in enumerate(zip(sample_texts, sample_actual)):
        pred_label, confidence = tester.predict_single(text)
        status = "✓" if actual == pred_label else "✗"
        print(
            f"{status} [{i + 1}] 商品: {text[:30]}... | 实际: {actual} | 预测: {pred_label} | 置信度: {confidence:.3f}")


if __name__ == "__main__":
    # 检查并安装必要库
    try:
        import torch
        import transformers
        import joblib
    except ImportError:
        print("正在安装必要库...")
        import subprocess

        subprocess.check_call(["pip", "install", "torch", "transformers", "joblib"])
        import torch
        import transformers
        import joblib

    main()