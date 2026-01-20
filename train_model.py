import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import  AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import joblib

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


class BertProductTrainer:
    """BERT商品分类训练器"""

    def __init__(self, model_name=r'D:\pythonAIclass\python2025Ai\TMF_Project\04-bert\bert-base-chinese', num_classes=10):
        self.model_name = model_name
        self.num_classes = num_classes
        print(f"正在下载/加载模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        self.model.to(device)
        self.label_encoder = LabelEncoder()
        self.history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

    def load_data(self, train_path):
        """加载训练数据"""
        train_data = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                train_data.append(data)

        return pd.DataFrame(train_data)

    def prepare_data(self, df):
        """准备模型输入数据"""
        texts = df['product_name'].tolist()
        labels = self.label_encoder.fit_transform(df['category'].tolist())
        return texts, labels

    def create_data_loader(self, texts, labels, batch_size=16, shuffle=True):
        """创建数据加载器"""
        dataset = ProductDataset(texts, labels, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, data_loader, optimizer, scheduler):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(data_loader, desc="Training")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(data_loader)

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

        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return accuracy, f1, predictions, true_labels

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """训练模型"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # 计算总步数
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_val_f1 = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)

            train_loss = self.train_epoch(train_loader, optimizer, scheduler)

            val_acc, val_f1, _, _ = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            print(f"训练损失: {train_loss:.4f}")
            print(f"验证准确率: {val_acc:.4f}")
            print(f"验证F1分数: {val_f1:.4f}")

            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model('model')
                print(f"保存最佳模型 (F1 = {val_f1:.4f})")

    def save_model(self, path):
        """保存模型"""
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # 保存标签编码器
        joblib.dump(self.label_encoder, f'{path}/label_encoder.pkl')


def main():
    print("初始化BERT商品分类训练器...")

    # 创建训练器实例
    trainer = BertProductTrainer(num_classes=10)

    print("正在加载训练数据...")
    train_df = trainer.load_data('data/train.jsonl')

    print(f"训练数据大小: {len(train_df)}")
    print(f"类别数量: {train_df['category'].nunique()}")

    # 准备数据
    print("准备训练数据...")
    train_texts, train_labels = trainer.prepare_data(train_df)

    # 分割验证集
    X_train, X_val, y_train, y_val = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42,
        stratify=train_labels
    )

    # 创建数据加载器
    train_loader = trainer.create_data_loader(X_train, y_train, batch_size=16)
    val_loader = trainer.create_data_loader(X_val, y_val, batch_size=16, shuffle=False)

    print(f"训练样本数: {len(X_train)}")
    print(f"验证样本数: {len(X_val)}")

    # 训练模型
    print("\n开始训练BERT模型...")
    trainer.train(train_loader, val_loader, epochs=3, learning_rate=2e-5)

    # 保存最终模型
    trainer.save_model('model')
    print(f"\n模型已保存到 'model' 文件夹")


if __name__ == "__main__":
    main()