import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from collections import Counter
from datetime import datetime

# loading data 
input_path =  ''
output_path = ''
df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./{current_time}"
os.makedirs(output_dir, exist_ok=True)

# Column processing and missing values
df.rename(columns={'Label': 'label'}, inplace=True)
df.columns = df.columns.str.strip()
print(df.columns)
if 'label' not in df.columns:
    print("⚠️ 未找到 'Label' 欄位")
    print(df.columns)
    exit()
df.fillna(0, inplace=True)


encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != 'label':
        df[col] = encoder.fit_transform(df[col])

# Separate features and labels
X_all = df.drop(columns=['label'])
y = df['label']


# For CICIDS 2017
# adasyn_features = [
#     'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
#     'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
#     'Fwd Packet Length Mean', 'Fwd Packet Length Std',
#     'Bwd Packet Length Mean', 'Bwd Packet Length Std',
#     'Flow Bytes/s', 'Flow Packets/s',
#     'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
#     'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
#     'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
#     'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
#     'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Average Packet Size',
#     'Active Mean', 'Active Std', 'Idle Mean', 'Idle Std'
# ]
# # for CICIOT 2023
# adasyn_features = [
#     'flow_duration', 'header_length', 'duration',
#     'rate', 'srate', 'drate',
#     'tot sum', 'min', 'max', 'avg', 'std', 'tot size',
#     'iat', 'number', 'magnitue', 'radius',
#     'covariance', 'variance', 'weight'
# ]

# For CICIDS 2019
adasyn_features = [
    'flow duration', 'total fwd packets', 'total backward packets',
    'total length of fwd packets', 'total length of bwd packets', 
    'fwd packet length max', 'fwd packet length min', 'fwd packet length mean',
    'fwd packet length std', 'bwd packet length max', 'bwd packet length min',
    'bwd packet length mean', 'bwd packet length std', 'flow bytes/s',
    'flow packets/s', 'flow iat mean', 'flow iat std', 'flow iat max',
    'flow iat min', 'fwd iat total', 'fwd iat mean', 'fwd iat std', 
    'fwd iat max', 'fwd iat min', 'bwd iat total', 'bwd iat mean',
    'bwd iat std', 'bwd iat max', 'bwd iat min', 'fwd psh flags',
    'fwd header length', 'bwd header length', 'fwd packets/s', 
    'bwd packets/s', 'min packet length', 'max packet length', 
    'packet length mean', 'packet length std', 'packet length variance', 
    'syn flag count', 'rst flag count', 'ack flag count', 'urg flag count',
    'cwe flag count', 'down/up ratio', 'average packet size', 
    'avg fwd segment size', 'avg bwd segment size', 'fwd header length.1',
    'subflow fwd packets', 'subflow fwd bytes', 'subflow bwd packets',
    'subflow bwd bytes', 'init_win_bytes_forward', 'init_win_bytes_backward', 
    'act_data_pkt_fwd', 'min_seg_size_forward', 'active mean', 'active std',
    'active max', 'active min', 'idle mean', 'idle std', 'idle max', 'idle min', 
    'inbound'
]


X = X_all[adasyn_features].copy()

# Cleaning and standardization
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)
X = np.clip(X, -1e6, 1e6)
X_scaled = StandardScaler().fit_transform(X)

# Category distribution and target balance value
label_count = Counter(y)
print("📊 Original category distribution:", label_count)

raw_median = int(np.median(list(label_count.values())))

target_count = raw_median

print(f"🌟 Targeted numbers: {target_count}")

undersample_df = []
adasyn_indices = []
no_action_df = []
sampling_strategy = {}

for label in label_count:
    class_indices = y[y == label].index
    class_count = label_count[label]

    if class_count > target_count * 1.5:
        min_allowed = int(class_count * 0.010)
        target = max(target_count, min_allowed)
        # target = target_count
        sampled_idx = np.random.choice(class_indices, size=target, replace=False)
        undersample_df.append(df.loc[sampled_idx])
        print(f"🔽 下採樣 {label}: {class_count} → {target} (不得低於5%)")
    elif class_count < target_count:
        max_allowed = class_count * 2
        adjusted_target = min(target_count, max_allowed)
        if adjusted_target > class_count:
            sampling_strategy[label] = adjusted_target
            adasyn_indices.extend(class_indices.tolist())
            print(f"🔼 將用 ADASYN 擴增 {label}: {class_count} → {adjusted_target} (不得高於原本的2倍)")
    else:
        no_action_df.append(df.loc[class_indices])
        print(f"⏸ 保留原樣 {label}: {class_count}")

balanced_df = pd.concat(undersample_df + no_action_df, axis=0)
intermediate_count = Counter(balanced_df['label'])

if sampling_strategy:
    X_ada = X_scaled[adasyn_indices]
    y_ada = y.iloc[adasyn_indices]

    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42, n_neighbors=3)
    X_resampled, y_resampled = adasyn.fit_resample(X_ada, y_ada)

    df_adasyn = pd.DataFrame(X_resampled, columns=adasyn_features)
    df_adasyn['label'] = y_resampled

    for col in X_all.columns:
        if col not in adasyn_features:
            mode_map = df.groupby('label')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
            df_adasyn[col] = df_adasyn['label'].map(mode_map)

    balanced_df = pd.concat([balanced_df, df_adasyn], axis=0)
    print(f"✅ ADASYN 擴增完成: {Counter(y_resampled)}")

balanced_df = balanced_df.sample(frac=1, random_state=42)
balanced_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 已儲存平衡資料至: {output_path}")

# ➕ Distribution analysis and image storage using PCA and t-SNE
print("📈 Begin PCA and t-SNE distribution analysis and image storage...")


label_encoder = LabelEncoder()
orig_labels = label_encoder.fit_transform(df['label'])
bal_labels = label_encoder.transform(balanced_df['label'])
label_names = label_encoder.classes_
colors = plt.cm.get_cmap('tab10', len(label_names))

# ---------- PCA ----------
X_pca_all = df[adasyn_features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_pca_all = np.clip(X_pca_all, -1e6, 1e6)
X_pca_scaled = StandardScaler().fit_transform(X_pca_all)

X_pca_balanced = balanced_df[adasyn_features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_pca_balanced = np.clip(X_pca_balanced, -1e6, 1e6)
X_balanced_scaled = StandardScaler().fit_transform(X_pca_balanced)

pca_orig = PCA(n_components=2).fit_transform(X_pca_scaled)
pca_bal = PCA(n_components=2).fit_transform(X_balanced_scaled)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for i, name in enumerate(label_names):
    axs[0].scatter(pca_orig[orig_labels == i, 0], pca_orig[orig_labels == i, 1], s=2, label=name, color=colors(i), alpha=0.5)
    axs[1].scatter(pca_bal[bal_labels == i, 0], pca_bal[bal_labels == i, 1], s=2, label=name, color=colors(i), alpha=0.5)
axs[0].set_title('PCA: Original Dataset')
axs[0].set_xlabel('PCA 1')
axs[0].set_ylabel('PCA 2')
axs[1].set_title('PCA: Balanced Dataset')
axs[1].set_xlabel('PCA 1')
axs[1].set_ylabel('PCA 2')
fig.legend(label_names, loc='center right', fontsize=8)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(os.path.join(output_dir, "pca_distribution_comparison.png"), dpi=300)
print("📸 PCA 分佈圖已儲存為: pca_distribution_comparison.png")

# ---------- t-SNE ----------
from sklearn.manifold import TSNE


sample_size = 3000
original_sample = df.sample(n=sample_size, random_state=42)
balanced_sample = balanced_df.sample(n=sample_size, random_state=42)
# sample_size = 0.1
# original_sample = df.groupby('Label').apply(lambda x: x.sample(frac=sample_size, random_state=42)).reset_index(drop=True)
# balanced_sample = balanced_df.groupby('Label').apply(lambda x: x.sample(frac=sample_size, random_state=42)).reset_index(drop=True)


X_original_tsne = original_sample[adasyn_features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_original_tsne = np.clip(X_original_tsne, -1e6, 1e6)
X_original_scaled = StandardScaler().fit_transform(X_original_tsne)

X_balanced_tsne = balanced_sample[adasyn_features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_balanced_tsne = np.clip(X_balanced_tsne, -1e6, 1e6)
X_balanced_scaled = StandardScaler().fit_transform(X_balanced_tsne)

original_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_original_scaled)
balanced_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_balanced_scaled)

sample_orig_labels = label_encoder.transform(original_sample['label'])
sample_bal_labels = label_encoder.transform(balanced_sample['label'])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for i, name in enumerate(label_names):
    axs[0].scatter(original_tsne[sample_orig_labels == i, 0], original_tsne[sample_orig_labels == i, 1], s=2, label=name, color=colors(i), alpha=0.5)
    axs[1].scatter(balanced_tsne[sample_bal_labels == i, 0], balanced_tsne[sample_bal_labels == i, 1], s=2, label=name, color=colors(i), alpha=0.5)
axs[0].set_title('t-SNE: Original Sampled')
axs[0].set_xlabel('t-SNE 1')
axs[0].set_ylabel('t-SNE 2')
axs[1].set_title('t-SNE: Balanced Sampled')
axs[1].set_xlabel('t-SNE 1')
axs[1].set_ylabel('t-SNE 2')
fig.legend(label_names, loc='center right', fontsize=8)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(os.path.join(output_dir, "tsne_distribution_comparison.png"), dpi=300)
print("📸 t-SNE distribution map has been stored as: tsne_distribution_comparison.png")
