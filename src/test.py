import nemo.collections.asr as nemo_asr
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import json
import IPython.display as ipd
import numpy as np
from sklearn.preprocessing import label_binarize


@torch.no_grad()
def extract_logits(model, dataloader):
    logits_buffer = []
    label_buffer = []

    # Follow the above definition of the test_step
    for batch in dataloader:
        audio_signal, audio_signal_len, labels, labels_len = batch
        # print(batch)
        # break
        logits = model(input_signal=audio_signal, input_signal_length=audio_signal_len)
        print(logits.shape)
        # print(logits)

        logits_buffer.append(logits)
        label_buffer.append(labels)
        print(".", end='')
    print()

    print("Finished extracting logits !")
    logits = torch.cat(logits_buffer, 0)
    labels = torch.cat(label_buffer, 0)
    return logits, labels


if __name__ == '__main__':
    vad_model = nemo_asr.models.EncDecClassificationModel.restore_from('nemo_experiments\marblenet\\2024-08-02_09-39-12\checkpoints\marblenet.nemo')

    checkpoint_path = 'nemo_experiments/marblenet/2024-08-02_09-39-12/checkpoints/marblenet--val_loss=0.0101-epoch=35.ckpt'
    checkpoint = torch.load(checkpoint_path)
    vad_model.load_state_dict(checkpoint['state_dict'], strict=False)

    config = vad_model.hparams 
    # config.cfg.test_ds.manifest_filepath='old_data/our_manifest/balanced_noise_testing_manifest.json,old_data/our_manifest/balanced_speech_testing_manifest.json' #'old_data\our_manifest\\fine_tune\\test_kaggle.json'
    vad_model.setup_test_data(config.cfg.test_ds)
    test_dl = vad_model._test_dl

    cpu_model = vad_model.cpu()
    cpu_model.eval()

    logits, labels = extract_logits(cpu_model, test_dl)

    probs = F.softmax(logits, dim=-1)
    # print(probs)
    predictions = torch.argmax(probs, dim=-1)
    # print(predictions)
    print("Logits:", logits.shape, "Labels :", labels.shape)

    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions_proba = probs.cpu().numpy()
    positive_class_probs = probs[:, 1]

    conf_matrix = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # ROC 커브 그리기 (이진 분류)

    fpr, tpr, _ = roc_curve(labels, positive_class_probs)  # 긍정 클래스의 확률 사용
    roc_auc = roc_auc_score(labels, positive_class_probs)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc})')
    plt.plot([0, 1], [0, 1], 'k--')  # 대각선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    #막대그래프

    # 메트릭 데이터
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # 막대그래프를 그리기 위한 데이터
    labels = list(metrics.keys())
    values = list(metrics.values())

    # 그래프 설정
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow'])

    # 그래프 제목과 레이블 설정
    plt.title('Performance Metrics', fontsize=16)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)  # 메트릭 값 범위 설정

    # 값 레이블 추가
    for i, value in enumerate(values):
        plt.text(i, value + 0.005, f'{value:.4f}', ha='center', fontsize=12)

    # 그래프 표시
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


