# GTSRB Traffic Sign Classifier

## 1. Mục tiêu
- Phân loại 43 loại biển báo giao thông (GTSRB).
- Ứng dụng: hỗ trợ hệ thống lái tự động/ADAS.

## 2. Dữ liệu
- Dataset: GTSRB (Kaggle).
- Số lớp: 43; Ảnh đa dạng điều kiện ánh sáng, góc nhìn.
- Tiền xử lý: resize 224x224, normalize; augment: rotate, shift, color jitter.

## 3. Mô hình
- Transfer learning: ResNet18 (pretrained ImageNet).
- Head: Linear + ReLU + Dropout + Linear (43 classes).

## 4. Huấn luyện
- Optim: AdamW, lr=3e-4; Scheduler: Cosineanneal; Loss: CrossEntropy.
- Mixed precision (amp) nếu có GPU.
- Early-save best model theo val accuracy.

## 5. Kết quả
- Val accuracy: ~97.5% (10 epoch).
- Confusion matrix, classification report.

## 6. Cách chạy
- `pip install -r requirements.txt`
- Tải data vào `data/GTSRB/` tại `https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data`
- Train: `python src/train.py`
- Eval: `python src/eval.py`
- Infer: `python src/infer.py --img ...`