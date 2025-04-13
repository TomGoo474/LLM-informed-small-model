import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import os

# 创建输出文件夹
output_dir = 'output_images_single_modality'
os.makedirs(output_dir, exist_ok=True)

# 定义常量
CHANNELS_INITIAL = 16
CHANNELS_RES = 64
CHANNELS_CONCAT = CHANNELS_RES * 2
CHANNELS_DECODER_CONCAT = CHANNELS_INITIAL * 2

# 数据加载部分（移除文本特征相关部分）
try:
    train_spectra_data = pd.read_excel('Train_Spectra.xlsx')
    test_spectra_data = pd.read_excel('Test_Spectra.xlsx')
except FileNotFoundError as e:
    print(f"Error: Data file not found! {e}")
    exit(1)

train_spectra_data = train_spectra_data.set_index('ID')
test_spectra_data = test_spectra_data.set_index('ID')

for spectra_data in [train_spectra_data, test_spectra_data]:
    if spectra_data['Class'].dtype == object:
        if spectra_data['Class'].str.match(r'^\d+$').all():
            spectra_data['Class'] = spectra_data['Class'].astype(int)
        else:
            class_mapping = {val: idx for idx, val in enumerate(spectra_data['Class'].unique())}
            spectra_data['Class'] = spectra_data['Class'].map(class_mapping)
            print(f"Class mapping for {spectra_data}:", class_mapping)
    else:
        spectra_data['Class'] = spectra_data['Class'].astype(int)

class SeedSpectralDataset(Dataset):
    def __init__(self, spectra_data):
        spectra = spectra_data.iloc[:, 1:].values
        self.spectra = torch.FloatTensor((spectra - spectra.mean()) / spectra.std())
        self.labels = torch.LongTensor(spectra_data['Class'].values)
        print(f"Spectra shape: {self.spectra.shape}, Min: {self.spectra.min():.4f}, Max: {self.spectra.max():.4f}")

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return {
            'spectra': self.spectra[idx],
            'label': self.labels[idx]
        }

train_dataset = SeedSpectralDataset(train_spectra_data)
test_dataset = SeedSpectralDataset(test_spectra_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResConv1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        return self.relu(out)

class ResidualLSTM(nn.Module):
    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM = nn.LSTM(d_model, d_model, num_layers=1, bidirectional=False)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        res = x
        x, _ = self.LSTM(x)
        x = self.linear(x)
        return F.relu(res + x)

class SingleModalityConvResModel(nn.Module):
    def __init__(self, spectral_dim, latent_dim, num_classes=2):
        super(SingleModalityConvResModel, self).__init__()
        self.spectral_dim = spectral_dim
        self.latent_dim = latent_dim

        # 编码器
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, CHANNELS_INITIAL, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(CHANNELS_INITIAL),
            nn.ReLU()
        )
        self.initial_lstm = ResidualLSTM(d_model=CHANNELS_INITIAL)
        self.resconv3 = ResConv1D(CHANNELS_INITIAL, CHANNELS_RES, kernel_size=3, stride=1, padding=1)
        self.resconv5 = ResConv1D(CHANNELS_INITIAL, CHANNELS_RES, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.pre_transformer_conv = nn.Conv1d(CHANNELS_CONCAT, 256, kernel_size=1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True),
            num_layers=2
        )
        self.transformer_fc = nn.Linear(256 * (spectral_dim // 2), latent_dim)

        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, CHANNELS_CONCAT * (spectral_dim // 2))
        self.decoder_resconv3 = ResConv1D(CHANNELS_RES, CHANNELS_INITIAL, kernel_size=3, stride=1, padding=1)
        self.decoder_resconv5 = ResConv1D(CHANNELS_RES, CHANNELS_INITIAL, kernel_size=5, stride=1, padding=2)
        self.decoder_transpose = nn.Sequential(
            nn.ConvTranspose1d(CHANNELS_DECODER_CONCAT, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, spectra):
        spectra = spectra.unsqueeze(1)  # (batch_size, 1, spectral_dim)
        x = self.initial_conv(spectra)
        x = x.transpose(1, 2).transpose(0, 1)  # (spectral_dim, batch_size, CHANNELS_INITIAL)
        x = self.initial_lstm(x)
        x = x.transpose(0, 1).transpose(1, 2)  # (batch_size, CHANNELS_INITIAL, spectral_dim)

        out3 = self.resconv3(x)
        out5 = self.resconv5(x)
        encoded = torch.cat((out3, out5), dim=1)  # (batch_size, CHANNELS_CONCAT, spectral_dim)
        encoded = self.pool(encoded)  # (batch_size, CHANNELS_CONCAT, spectral_dim // 2)

        encoded = self.pre_transformer_conv(encoded)  # (batch_size, 256, spectral_dim // 2)
        encoded = encoded.transpose(1, 2)  # (batch_size, spectral_dim // 2, 256)
        transformer_out = self.transformer_encoder(encoded)
        transformer_flat = transformer_out.reshape(transformer_out.size(0), -1)
        spectral_latent = self.transformer_fc(transformer_flat)  # (batch_size, latent_dim)

        decoded = self.decoder_fc(spectral_latent)
        decoded = decoded.view(-1, CHANNELS_CONCAT, self.spectral_dim // 2)
        decoded_split = torch.split(decoded, CHANNELS_RES, dim=1)
        dec_out3 = self.decoder_resconv3(decoded_split[0])
        dec_out5 = self.decoder_resconv5(decoded_split[1])
        decoded_concat = torch.cat((dec_out3, dec_out5), dim=1)
        reconstruction = self.decoder_transpose(decoded_concat)  # (batch_size, 1, spectral_dim)
        reconstruction = reconstruction.squeeze(1)  # (batch_size, spectral_dim)

        class_logits = self.classifier(spectral_latent)  # (batch_size, num_classes)

        return spectral_latent, reconstruction, class_logits

# 损失函数（移除余弦相似性损失）
criterion_reconstruction = nn.MSELoss(reduction='none')
criterion_classification = nn.CrossEntropyLoss()

input_spectral_dim = train_spectra_data.iloc[:, 1:].shape[1]
latent_dim = 256

class DynamicLossWeights:
    def __init__(self, beta=0.5, gamma=0.5):
        self.beta = beta
        self.gamma = gamma

    def update(self, recon_loss, class_loss):
        total = recon_loss + class_loss
        self.beta = recon_loss / total
        self.gamma = class_loss / total
        norm = self.beta + self.gamma
        self.beta /= norm
        self.gamma /= norm

loss_weights = DynamicLossWeights()

def spectral_similarity(reconstruction, spectra):
    similarity = torch.cosine_similarity(reconstruction, spectra, dim=1)
    return similarity.mean().item()

def identify_key_wavelengths(reconstruction, spectra, top_k=5):
    mse_per_wavelength = criterion_reconstruction(reconstruction, spectra).mean(dim=0)
    key_wavelengths = torch.topk(mse_per_wavelength, k=top_k, largest=True).indices
    return key_wavelengths.cpu().numpy()

def visualize_spectra(spectra, reconstruction, epoch, run, prefix="train", num_samples=3, top_k=5):
    for i in range(min(num_samples, spectra.size(0))):
        plt.figure(figsize=(12, 6))
        original = spectra[i].detach().cpu().numpy()
        recon = reconstruction[i].detach().cpu().numpy()
        plt.plot(original, label="Original", color='blue')
        plt.plot(recon, label="Reconstructed", color='red', linestyle='--')
        key_wavelengths = identify_key_wavelengths(reconstruction[i:i+1], spectra[i:i+1], top_k=top_k)
        for wl in key_wavelengths:
            plt.axvline(x=wl, color='green', linestyle=':', alpha=0.5, label=f"Key WL {wl}" if wl == key_wavelengths[0] else None)
        plt.title(f"{prefix.capitalize()} Spectra - Run {run + 1}, Epoch {epoch + 1}, Sample {i + 1}")
        plt.xlabel("Wavelength Index")
        plt.ylabel("Intensity")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{prefix}_spectra_run{run + 1}_epoch{epoch + 1}_sample{i + 1}.png"))
        plt.close()

def plot_confusion_matrix(y_true, y_pred, run):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Vigor', 'Non-vigor'], yticklabels=['Vigor', 'Non-vigor'])
    plt.title(f"Confusion Matrix - Run {run + 1}, Accuracy: {acc:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_run{run + 1}.png"))
    plt.close()
    return acc

def train_and_evaluate(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SingleModalityConvResModel(input_spectral_dim, latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    num_epochs = 90
    best_val_acc = 0.0
    best_model_path = f'main2_best_model_run_{seed}.pth'

    train_recon_losses, test_recon_losses = [], []
    train_spectral_accs, test_spectral_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = train_acc = train_recon_loss = train_spectral_acc = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Run {seed - 41} Epoch {epoch + 1}/{num_epochs}")):
            spectra = batch['spectra'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            spectral_latent, reconstruction, class_logits = model(spectra)

            loss_reconstruction = criterion_reconstruction(reconstruction, spectra).mean()
            loss_classification = criterion_classification(class_logits, labels)
            loss_weights.update(loss_reconstruction.item(), loss_classification.item())
            loss = (loss_weights.beta * loss_reconstruction + loss_weights.gamma * loss_classification)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += loss_reconstruction.item()
            train_spectral_acc += spectral_similarity(reconstruction, spectra)
            preds = torch.argmax(class_logits, dim=1)
            train_acc += (preds == labels).float().mean().item()

            if epoch % 50 == 0 and batch_idx == 0:
                visualize_spectra(spectra, reconstruction, epoch, seed - 42, prefix="train")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_spectral_acc = train_spectral_acc / len(train_loader)
        train_recon_losses.append(avg_train_recon_loss)
        train_spectral_accs.append(avg_train_spectral_acc)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Recon Loss: {avg_train_recon_loss:.4f}, "
              f"Spectral Acc: {avg_train_spectral_acc:.4f}, Class Acc: {avg_train_acc:.4f}")

        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            test_recon_loss = test_spectral_acc = 0.0
            for batch in test_loader:
                spectra = batch['spectra'].to(device)
                labels = batch['label'].to(device)

                spectral_latent, reconstruction, class_logits = model(spectra)
                loss_reconstruction = criterion_reconstruction(reconstruction, spectra).mean()
                test_recon_loss += loss_reconstruction.item()
                test_spectral_acc += spectral_similarity(reconstruction, spectra)
                preds = torch.argmax(class_logits, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            current_val_acc = accuracy_score(y_true, y_pred)
            avg_test_recon_loss = test_recon_loss / len(test_loader)
            avg_test_spectral_acc = test_spectral_acc / len(test_loader)
            test_recon_losses.append(avg_test_recon_loss)
            test_spectral_accs.append(avg_test_spectral_acc)
            print(f"Val Acc: {current_val_acc:.4f}, Test Recon Loss: {avg_test_recon_loss:.4f}, "
                  f"Spectral Acc: {avg_test_spectral_acc:.4f}")

            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            spectra = batch['spectra'].to(device)
            labels = batch['label'].to(device)
            _, _, class_logits = model(spectra)
            preds = torch.argmax(class_logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm_acc = plot_confusion_matrix(y_true, y_pred, seed - 42)
    print(f"Confusion Matrix Accuracy for Run {seed - 41}: {cm_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc, train_recon_losses, test_recon_losses, train_spectral_accs, test_spectral_accs

num_runs = 10
val_accuracies = []
all_train_recon_losses, all_test_recon_losses = [], []
all_train_spectral_accs, all_test_spectral_accs = [], []

for run in range(num_runs):
    print(f"\nStarting Run {run + 1}/{num_runs}")
    seed = 42 + run
    best_acc, train_recon_losses, test_recon_losses, train_spectral_accs, test_spectral_accs = train_and_evaluate(seed)
    val_accuracies.append(best_acc)
    all_train_recon_losses.append(train_recon_losses)
    all_test_recon_losses.append(test_recon_losses)
    all_train_spectral_accs.append(train_spectral_accs)
    all_test_spectral_accs.append(test_spectral_accs)
    print(f"Run {run + 1} completed with Best Val Acc: {best_acc:.4f}")

# 统计结果
print("\nDetailed Results for Each Run:")
for i, acc in enumerate(val_accuracies):
    print(f"Run {i + 1}: Best Val Acc = {acc:.4f}")

mean_acc = statistics.mean(val_accuracies)
std_acc = statistics.stdev(val_accuracies) if len(val_accuracies) > 1 else 0
max_acc = max(val_accuracies)
min_acc = min(val_accuracies)

mean_train_recon_loss = [statistics.mean([run_losses[epoch] for run_losses in all_train_recon_losses])
                         for epoch in range(len(all_train_recon_losses[0]))]
mean_test_recon_loss = [statistics.mean([run_losses[epoch] for run_losses in all_test_recon_losses])
                        for epoch in range(len(all_test_recon_losses[0]))]
mean_train_spectral_acc = [statistics.mean([run_accs[epoch] for run_accs in all_train_spectral_accs])
                           for epoch in range(len(all_train_spectral_accs[0]))]
mean_test_spectral_acc = [statistics.mean([run_accs[epoch] for run_accs in all_test_spectral_accs])
                          for epoch in range(len(all_test_spectral_accs[0]))]

print("\nFinal Statistics:")
print(f"Mean Acc: {mean_acc:.4f}, Std: {std_acc:.4f}, Max: {max_acc:.4f}, Min: {min_acc:.4f}")
print(f"Mean Train Recon Loss: {mean_train_recon_loss[-1]:.4f}")
print(f"Mean Test Recon Loss: {mean_test_recon_loss[-1]:.4f}")
print(f"Mean Train Spectral Acc: {mean_train_spectral_acc[-1]:.4f}")
print(f"Mean Test Spectral Acc: {mean_test_spectral_acc[-1]:.4f}")

results = {
    'accuracies': val_accuracies,
    'mean_acc': mean_acc,
    'std_acc': std_acc,
    'max_acc': max_acc,
    'min_acc': min_acc,
    'train_recon_losses': all_train_recon_losses,
    'test_recon_losses': all_test_recon_losses,
    'train_spectral_accs': all_train_spectral_accs,
    'test_spectral_accs': all_test_spectral_accs,
    'mean_train_recon_loss': mean_train_recon_loss,
    'mean_test_recon_loss': mean_test_recon_loss,
    'mean_train_spectral_acc': mean_train_spectral_acc,
    'mean_test_spectral_acc': mean_test_spectral_acc
}
np.save('single_modality_results.npy', results)