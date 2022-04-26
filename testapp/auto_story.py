'''
!pip install git+https://github.com/takuseno/d4rl-pybullet
!pip install typed-argument-parser
!apt update
!apt install xvfb
!pip install gym-notebook-wrapper
'''
#import os

#os.chdir(work_dir) #カレントディレクトリをここにするやつ
#print(os.getcwd())

import random
import math
import numpy as np
from os.path import join
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from testapp.world_model_lec6.utils_language import Trainer as LanguageTrainer
from testapp.world_model_lec6.utils_language import TrainerConfig as LTConfig
from testapp.world_model_lec6.utils_language import sample

#import gym
#from gym.envs.registration import register
#import d4rl_pybullet
#import gnwrapper
#from IPython.display import HTML
from base64 import b64encode
from testapp.world_model_lec6.utils_rl import (
    #Config, DiscretizedDataset,
    load_model, load_from_config, load_environment,
    filter_cdf, crop_x, update_context,
)

# 再現性のためシードを設定します.
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.set_printoptions(edgeitems=1e3)

DEVICE = 'cpu'

# 要素にドットでアクセスできる辞書クラス
class Args(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

#Multi-Head Attention
class CausalSelfAttention(nn.Module):

    def __init__(self, config, resid_pdrop=0.1, attn_pdrop=0.1):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # 入力をK, Q, Vにそれぞれ変換する全結合層.
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # ドロップアウト正則化
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # Multi-Head Attentionアウトプットの全結合層
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # torch.trilは行列の右上三角部分をゼロにして返します（予測するトークンの右側をマスク）
        # nn.Moduleのregister_bufferは, モデルのパラメータとならないtensorを追加するのに使われます.
        self.register_buffer(name="mask",
                tensor=torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        
        if config.rl: # 後半の強化学習のための処理を行うとき用のフラグ
            joined_dim = config.observation_dim + config.action_dim + 2
            # それまでのvalue estimatesをマスクする
            self.mask.squeeze()[:, joined_dim-1::joined_dim] = 0 # (block_size, block_size)

    def forward(self, x, layer_past=None):
        # B: バッチサイズ (batch_size)
        # T: シークエンスの長さ. コンテクストサイズ （block_size, または上のn）よりも小さくないといけない.
        # C: Embedding空間の次元数. 上の説明のd_modelに対応
        B, T, C = x.size()

        # Key, Que, Valueをそれぞれの全結合層で計算
        
        k = self.key(x) # WRITE ME # ( B, T, C )
        q = self.query(x) # WRITE ME # ( B, T, C )
        v = self.value(x) # WRITE ME # ( B, T, C )

        # Multi-Head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # ( B, n_heads, T, d_k )
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # ( B, n_heads, T, d_k )
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # ( B, n_heads, T, d_v )

        # QとKの行列積をとり, sqrt(d_k)でスケール
        # ( B, n_heads, T, d_k ) x ( B, n_heads, d_k, T ) -> ( B, n_heads, T, T )

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #(q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)) # WRITE ME

        # マスク
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1) # ( B, n_heads, T, T )

        # 正則化
        att = self.attn_drop(att)
        # VとのMatMul
        # ( B, n_heads, T, T ) x ( B, n_heads, T, d_v ) -> ( B, n_heads, T, d_v )

        y = att @ v # WRITE ME

        # 各headからの出力を結合する.
        # ( B, n_heads, T, d_v ) -> ( B, T, n_heads, d_v ) -> ( B, T, embd_dim )
        # contiguousはviewするためにtensorの要素をメモリ上で並べ直すもの
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Attentionアウトプットの全結合層
        y = self.resid_drop(self.proj(y)) # ( B, T, embd_dim )

        return y

#ブロックの定義
class Block(nn.Module):

    def __init__(self, config, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        # Feed Forward
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            # GELU(Gaussian Error Linear Units)活性化関数. 形としてはReLUに似ている.
            # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        # AttentionやFeedForwardの前にLayerNormが移動している.
        # x自身を足すことでresidual connection
        
        x = self.attn(self.ln1(x))+x # WRITE ME
        x = self.mlp(self.ln2(x))+x # WRITE ME
        return x

#GPTmodel
class GPT(nn.Module):

    def __init__(self, config, embd_pdrop=0.1):
        super().__init__()

        # 文字のidx（整数）と表現（ベクター）をつなぐルックアップテーブル
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Positional encodingで足すベクターはゼロで初期化し, 学習可能なnn.Parameterとする
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # 確率0.1でランダムにtensorの要素をゼロにする
        self.drop = nn.Dropout(embd_pdrop)
        # n_layer個のブロックをnn.Sequentialで繋げる
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # デコーダのhead
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def forward(self, idx, targets=None): # idx: ( B, T ), targets: ( B, T )
        # targetsはidxの全要素が一個ずつ前にずれたもの.
        b, t = idx.size()
        # 入力シークエンスがコンテクストサイズを超えていないことを保証する
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # 順伝播
        # 文字のidx（整数）列を表現（ベクター）に変換
        token_embeddings = self.tok_emb(idx) # ( B, T, embd_dim )
        position_embeddings = self.pos_emb[:, :t, :] # ( 1, T, embd_dim )
        # トークンベクターに位置情報ベクターを足し, ドロップアウトをかけます.

        x = self.drop(token_embeddings + position_embeddings) # WRITE ME # ( B, T, embd_dim )

        x = self.blocks(x)
        # GPT-2で追加された最後のlayer normalization
        x = self.ln_f(x) # ( B, T, embd_dim )

        logits = self.head(x) # ( B, T, vocab_size )

        # 訓練のときロス計算
        if targets is not None:
            # cross_entropy{ ( B*T, vocab_size ), ( B*T, ) } ->  ( B*T, )
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None

        # 訓練のときは, lossを使ってtrainerがbackprop→パラメータ更新を行う.
        # サンプリングのときは, logitsを使ってtopkが計算される.
        return logits, loss


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        各種パラメータを, AdamWによる更新式にL2正則化項を加えるものと加えないもの
        (biases, layer norm / embedding weights)の2グループに分け, 最後に
        PyTorchのoprimizerを返しています. あまり重要ではないので理解する必要はありません.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)

        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # biasはweight decayされない
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # whitelistに入ったパラメータはweight decayされる
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # blacklistに入ったパラメータはweight decayされない
                    no_decay.add(fpn)
        # positional embeddingのパラメータはweight decayされない
        no_decay.add('pos_emb')

        # 見過ごされたパラメータが無いかチェックする.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

#文章生成
class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data))) # set型は重複しない要素のコレクション
        data_size, vocab_size = len(data), len(chars) # 1115394, 65
        print('Data has %d characters, %d unique:' % (data_size, vocab_size))
        print(chars)
        
        # 文字を0~64の数字にマップする辞書
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        # 逆に, 数字を65種類の文字にマップする辞書
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size # 65
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # dataからblock_size+1の長さの文字列を抜き出してくる.
        chunk = self.data[idx:idx + self.block_size + 1] # 例: 'First, '
        # 抜き出した文字列を一つ一つ数字にマップする
        dix = [self.stoi[s] for s in chunk] # 例: [18, 47, 56, 57, 58, 1, ...]

        # x, yともにblock_size. yはxの全要素が一個ずつ前にずれたもの.
        # モデルにはそれぞれidx, targetsとして渡される.
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

'''
block_size = 128 # コンテクストの長さ
dire = os.getcwd()
work_dir = dire+'/testapp/world_model_lec6/'
# 事前学習用データセット. ファイルは1.1MB程度です.
text = open(work_dir+'shakespeare.txt', 'r').read()
train_dataset = CharDataset(text, block_size)
args = Args({
        'vocab_size': train_dataset.vocab_size,
        'block_size': train_dataset.block_size,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 512,
        'rl': False, # Trajectory Transformerで行う処理を飛ばす
    })
model=GPT(args) # あとでtrainerがモデルをGPUに移してくれます.
    # LanguageTrainerをインスタンス化し, 訓練を開始します.
    # batch_sizeを512にするとメモリ不足になります.
tconf = LTConfig(max_epochs=2, batch_size=256, learning_rate=6e-4,
                lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                num_workers=2)
trainer = LanguageTrainer(model, train_dataset, None, tconf)
'''
def My_story(model, train_dataset, trainer,Input='This is input', Work_dir='', Top_k=10, GPU=True, STEPS=1000):
    model_path = Work_dir + '/trained_language_model.pth'
    # (英語で)自由に書いてみてください. 続きを生成してくれます.
    context = Input #"FIGHT:" #"MASTER YODA:"

    if not GPU:
        # 学習済みパラメータのロード
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    else:
         model.load_state_dict(torch.load(model_path))

    # 文字列を整数（インデックス）の配列にする
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    # xを学習済みモデルに入力し, 続きをサンプリングする
    y = sample(model, x, steps=STEPS, sample=True, top_k=Top_k)[0]

    # 整数（インデックス）の配列を文字列に戻す
    completion = ''.join([train_dataset.itos[int(i)] for i in y])

    # CUDA out of memory を防ぐ
    #torch.cuda.empty_cache()
    return completion