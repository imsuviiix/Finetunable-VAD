import nemo.collections.asr as nemo_asr
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

@torch.no_grad()
def extract_logits(model, dataloader):
    logits_buffer = []
    label_buffer = []

    # Follow the above definition of the test_step
    for batch in dataloader:
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = model(input_signal=audio_signal, input_signal_length=audio_signal_len)

        logits_buffer.append(logits)
        label_buffer.append(labels)
        print(".", end='')
    print()

    print("Finished extracting logits !")
    logits = torch.cat(logits_buffer, 0)
    labels = torch.cat(label_buffer, 0)
    return logits, labels

if __name__ == '__main__':
    vad_model = nemo_asr.models.EncDecClassificationModel.restore_from('nemo_experiments\marblenet\\2024-07-19_08-58-13\checkpoints\marblenet.nemo')

    checkpoint_path = 'nemo_experiments\marblenet\\2024-07-19_08-58-13\checkpoints\marblenet--val_loss=0.0021-epoch=17.ckpt'
    checkpoint = torch.load(checkpoint_path)
    vad_model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    config = vad_model.hparams 
    
    vad_model.setup_test_data(config.cfg.test_ds)
    test_dl = vad_model._test_dl

    cpu_model = vad_model.cpu()
    cpu_model.eval()

    logits, labels = extract_logits(cpu_model, test_dl)

    probs = F.softmax(logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1)

    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    conf_matrix = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")