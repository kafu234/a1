import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import sacrebleu
from tqdm import tqdm
import math
import os  # 用于检查分词器文件
import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, max_seq_length, d_model = x.size()
        return x.view(batch_size, max_seq_length, self.n_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))

        return output


class ManualSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        super(ManualSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 定义 window_size 为 "半径"
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, max_seq_length, d_model = x.size()
        return x.view(batch_size, max_seq_length, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V 形状: [batch_size, n_heads, seq_length, d_k]

        # N_q 和 N_k 分别是 Q 和 K 的序列长度
        # (在自注意力中 N_q == N_k, 在交叉注意力中可能不同)
        N_q = Q.size(-2)
        N_k = K.size(-2)

        # 1. (内存 O(N^2)) - 计算完整的注意力分数
        # attn_scores 形状: [batch_size, n_heads, N_q, N_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. (核心) - 创建并应用滑动窗口掩码 (Sliding Window Mask)

        # 创建 [N_q, 1] 和 [1, N_k] 的索引
        q_indices = torch.arange(N_q, device=Q.device).view(N_q, 1)
        k_indices = torch.arange(N_k, device=K.device).view(1, N_k)

        # 计算相对索引 ( [N_q, N_k] 矩阵)
        # rel_indices[i, j] = j - i
        rel_indices = k_indices - q_indices

        # 根据半径创建窗口掩码: -w <= (j - i) <= w
        # window_mask 形状: [N_q, N_k] (布尔型)
        window_mask = (rel_indices >= -self.window_size) & \
                      (rel_indices <= self.window_size)

        # 添加广播维度，变为 [1, 1, N_q, N_k]
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)

        # 应用窗口掩码：在窗口 *之外* 的位置设为 -infinity
        attn_scores = attn_scores.masked_fill(~window_mask, -1e9)

        # 3. 应用原始的 padding/causal 掩码
        if mask is not None:
            # mask (来自 generate_mask) 是 0/1 (或 True/False)
            # 0 (False) 的位置是需要被掩盖的
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 4. 计算 Softmax 和 V 的加权平均
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q, K, V, mask=None):
        # 1. 线性投射 (保持不变)
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 2. 分割头 (保持不变)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 计算注意力 (现在使用我们新的、带窗口掩码的版本)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. 组合头 (保持不变)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class Positionembedding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(Positionembedding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.FC1 = nn.Linear(d_model, d_ff)
        self.FC2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.FC2(self.relu(self.FC1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # self.attn = MultiHeadAttention(d_model, n_heads)
        self.attn = ManualSparseAttention(d_model, n_heads, window_size=64)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # self.self_attn = MultiHeadAttention(d_model, n_heads)
        # self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn = ManualSparseAttention(d_model, n_heads, window_size=64)
        self.cross_attn = ManualSparseAttention(d_model, n_heads, window_size=64)

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(self.dropout(self_attn) + x)
        cross_attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(self.dropout(cross_attn) + x)

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tar_vocab_size, d_model, n_heads, d_ff, max_seq_length, dropout, num_layers):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tar_vocab_size, d_model)
        self.positionembedding = Positionembedding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tar_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)

        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embeded = self.dropout(self.positionembedding(self.encoder_embedding(src)))
        enc_output = src_embeded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        tgt_embeded = self.dropout(self.positionembedding(self.decoder_embedding(tgt)))
        dec_output = tgt_embeded

        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

D_MODEL = 512
N_HEADS = 8
NUM_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1
MAX_SEQ_LENGTH = 128

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 15

DATASET_NAME = "iwslt2017"
DATASET_CONFIG = "iwslt2017-en-de"  # 英语到德语
SRC_LANG = "en"
TGT_LANG = "de"
SRC_TOKENIZER_FILE = "tokenizer_en.json"
TGT_TOKENIZER_FILE = "tokenizer_de.json"

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


def translate(model, src_text, src_tokenizer, tgt_tokenizer, device, max_length=MAX_SEQ_LENGTH):
    """
    使用 Greedy Decode 执行翻译
    """
    model.eval()

    # --- 1. 准备源序列 ---
    # 截断以留出 SOS/EOS 空间
    src_ids_raw = src_tokenizer.encode(src_text, add_special_tokens=False).ids[:max_length - 2]
    # 添加 SOS 和 EOS
    src_ids = [SRC_SOS_ID] + src_ids_raw + [SRC_EOS_ID]
    # 转换为 [1, seq_len] 的批次
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    # --- 2. 运行编码器 ---
    # 创建源掩码
    src_mask = (src_tensor != SRC_PAD_ID).unsqueeze(1).unsqueeze(2)

    # 手动执行编码器前向传播
    src_embeded = model.dropout(model.positionembedding(model.encoder_embedding(src_tensor)))
    enc_output = src_embeded
    for enc_layer in model.encoder_layers:
        enc_output = enc_layer(enc_output, src_mask)

    # --- 3. 运行解码器 (Greedy) ---
    # 从 [SOS] 标记开始
    tgt_ids = [TGT_SOS_ID]

    for _ in range(max_length - 1):  # -1 因为我们已经有 SOS
        # 转换为 [1, current_seq_len]
        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)

        # --- 创建目标掩码 ---
        # 1. 目标 Padding 掩码 (虽然我们没有 padding, 但模型需要这个维度)
        tgt_pad_mask = (tgt_tensor != TGT_PAD_ID).unsqueeze(1).unsqueeze(3)
        # 2. 目标 "No-Peek" (Causal) 掩码
        tgt_seq_len = tgt_tensor.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_seq_len, tgt_seq_len, device=device), diagonal=1)).bool()
        # 结合
        tgt_mask = tgt_pad_mask & nopeak_mask

        # --- 手动执行解码器前向传播 ---
        tgt_embeded = model.dropout(model.positionembedding(model.decoder_embedding(tgt_tensor)))
        dec_output = tgt_embeded
        for dec_layer in model.decoder_layers:
            # 这里的 src_mask 保持不变
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # --- 预测下一个词 ---
        # 传递通过最后的线性层
        output_logits = model.fc(dec_output)
        # 只关心序列中 *最后一个* 词的 logits
        last_token_logits = output_logits[:, -1, :]

        # 贪婪地选择最可能的词
        pred_token = torch.argmax(last_token_logits, dim=1).item()

        # 添加到我们的目标序列
        tgt_ids.append(pred_token)

        # 如果是 [EOS] 标记，则停止
        if pred_token == TGT_EOS_ID:
            break

    # --- 4. 解码为文本 ---
    # 解码 ID 序列，跳过特殊标记 (如 SOS, EOS, PAD)
    return tgt_tokenizer.decode(tgt_ids, skip_special_tokens=True)


def calculate_bleu_score(model, data_loader, src_tokenizer, tgt_tokenizer, device):
    print("Calculating BLEU score...")
    model.eval()
    candidates = [] # 候选句 (模型翻译)
    references = [] # 参考句 (人工翻译)
    
    # 我们遍历 .dataset 来获取原始文本，而不是批处理后的
    # 这对于逐句翻译和 BLEU 计算更简单
    for item in tqdm(data_loader.dataset, desc="Translating validation set"):
        src_text = item['translation'][SRC_LANG]
        ref_text = item['translation'][TGT_LANG]
        
        # 使用 no_grad 来节省内存和加速
        with torch.no_grad():
            candidate_text = translate(model, src_text, src_tokenizer, tgt_tokenizer, device)
        
        candidates.append(candidate_text)
        references.append(ref_text)
    
    # 计算 BLEU
    # sacrebleu 需要一个候选列表和 *一个* 参考列表
    #（如果每个候选有多个参考，则为 [refs1, refs2, ...]）
    # 在 IWSLT 中，我们只有一个参考，所以是 [references]
    bleu = sacrebleu.corpus_bleu(candidates, [references])
    
    return bleu.score


def get_text_iterator(dataset, lang):
    """用于分词器训练的文本迭代器"""
    for item in dataset:
        yield item['translation'][lang]


def train_tokenizer(dataset, lang, tokenizer_path):
    """训练并保存一个新的分词器"""
    tokenizer = Tokenizer(models.WordPiece(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=30000,
        special_tokens=SPECIAL_TOKENS
    )

    print(f"Training {lang} tokenizer...")
    tokenizer.train_from_iterator(get_text_iterator(dataset, lang), trainer=trainer)
    tokenizer.save(tokenizer_path)
    print(f"Saved {lang} tokenizer to {tokenizer_path}")
    return tokenizer


dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
train_data = dataset['train']
val_data = dataset['validation']

if os.path.exists(SRC_TOKENIZER_FILE):
    print("Loading source tokenizer...")
    src_tokenizer = Tokenizer.from_file(SRC_TOKENIZER_FILE)
else:
    src_tokenizer = train_tokenizer(train_data, SRC_LANG, SRC_TOKENIZER_FILE)

if os.path.exists(TGT_TOKENIZER_FILE):
    print("Loading target tokenizer...")
    tgt_tokenizer = Tokenizer.from_file(TGT_TOKENIZER_FILE)
else:
    tgt_tokenizer = train_tokenizer(train_data, TGT_LANG, TGT_TOKENIZER_FILE)

SRC_VOCAB_SIZE = src_tokenizer.get_vocab_size()
TGT_VOCAB_SIZE = tgt_tokenizer.get_vocab_size()

SRC_PAD_ID = src_tokenizer.token_to_id(PAD_TOKEN)
TGT_PAD_ID = tgt_tokenizer.token_to_id(PAD_TOKEN)
TGT_SOS_ID = tgt_tokenizer.token_to_id(SOS_TOKEN)
TGT_EOS_ID = tgt_tokenizer.token_to_id(EOS_TOKEN)

SRC_SOS_ID = src_tokenizer.token_to_id(SOS_TOKEN)
SRC_EOS_ID = src_tokenizer.token_to_id(EOS_TOKEN)

# 确保 PAD_ID 为 0, 以匹配你的掩码逻辑
assert SRC_PAD_ID == 0, f"Source PAD ID is {SRC_PAD_ID}, but model expects 0"
assert TGT_PAD_ID == 0, f"Target PAD ID is {TGT_PAD_ID}, but model expects 0"
print(f"Source Vocab Size: {SRC_VOCAB_SIZE}")
print(f"Target Vocab Size: {TGT_VOCAB_SIZE}")
print(f"Source PAD ID: {SRC_PAD_ID}")
print(f"Target PAD ID: {TGT_PAD_ID}")


def collate_fn(batch):
    src_batch, tgt_input_batch, tgt_output_batch = [], [], []

    # 截断长度（为 [SOS] 和 [EOS] 留出空间）
    TRUNC_LEN = MAX_SEQ_LENGTH - 2

    for item in batch:
        # --- 处理源序列 ---
        src_text = item['translation'][SRC_LANG]
        src_ids = src_tokenizer.encode(src_text, add_special_tokens=False).ids[:TRUNC_LEN]
        src_ids = [src_tokenizer.token_to_id(SOS_TOKEN)] + src_ids + [src_tokenizer.token_to_id(EOS_TOKEN)]
        # 填充
        src_padded = src_ids + [SRC_PAD_ID] * (MAX_SEQ_LENGTH - len(src_ids))
        src_batch.append(src_padded)

        # --- 处理目标序列 ---
        tgt_text = item['translation'][TGT_LANG]
        tgt_ids = tgt_tokenizer.encode(tgt_text, add_special_tokens=False).ids[:TRUNC_LEN]

        # 创建 decoder input (e.g., [SOS] + "Guten Tag")
        tgt_input_ids = [TGT_SOS_ID] + tgt_ids
        tgt_input_padded = tgt_input_ids + [TGT_PAD_ID] * (MAX_SEQ_LENGTH - len(tgt_input_ids))
        tgt_input_batch.append(tgt_input_padded)

        # 创建 target output (e.g., "Guten Tag" + [EOS])
        tgt_output_ids = tgt_ids + [TGT_EOS_ID]
        tgt_output_padded = tgt_output_ids + [TGT_PAD_ID] * (MAX_SEQ_LENGTH - len(tgt_output_ids))
        tgt_output_batch.append(tgt_output_padded)

    return (
        torch.tensor(src_batch, dtype=torch.long),
        torch.tensor(tgt_input_batch, dtype=torch.long),
        torch.tensor(tgt_output_batch, dtype=torch.long)
    )


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tar_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=DROPOUT,
    num_layers=NUM_LAYERS
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for src, tgt_input, tgt_output in progress_bar:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        output = model(src, tgt_input)

        loss = criterion(
            output.view(-1, TGT_VOCAB_SIZE),
            tgt_output.view(-1)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)
        for src, tgt_input, tgt_output in progress_bar:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            output = model(src, tgt_input)

            loss = criterion(
                output.view(-1, TGT_VOCAB_SIZE),
                tgt_output.view(-1)
            )
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


print("Starting training...")

train_loss_history = []
val_loss_history = []

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss = evaluate(model, val_loader, criterion, DEVICE)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    print(f"--- Epoch {epoch}/{EPOCHS} ---")
    print(f"\tTrain Loss: {train_loss:.4f}")
    print(f"\tVal Loss:   {val_loss:.4f}")
    print("-" * (17 + len(str(epoch)) + len(str(EPOCHS))))

print("Training complete.")

final_bleu_score = calculate_bleu_score(model, val_loader, src_tokenizer, tgt_tokenizer, DEVICE)

print(f"\n=====================================")
print(f"Final Validation BLEU Score: {final_bleu_score:.2f}")
print(f"=====================================")

# ... (BLEU 分数计算和打印) ...
# print(f"Final Validation BLEU Score: {final_bleu_score:.2f}")
# ...

# --- 8. 绘制 Loss 曲线 ---
print("Generating loss curve...")

# 创建 X 轴 (Epochs)
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_loss_history, 'b-o', label='Training Loss')
plt.plot(epochs_range, val_loss_history, 'r-o', label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (Cross Entropy)') 
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.xticks(epochs_range)  # 确保 X 轴显示整数 Epoch

# 将图表保存到文件
plt.savefig('loss_curve.png')

# 尝试显示图表 (在某些环境如服务器上可能无法显示)
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot automatically, but saved to 'loss_curve.png'. Error: {e}")

print("Loss curve saved to loss_curve.png")
