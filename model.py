import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BertModel,BertConfig
import pickle
import torch.nn.functional as F
from torch.autograd import Function


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()

        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Instrument', 'Pitch', 'Duration', 'Velocity', 'TimeSig', 'Tempo']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256] * 8
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.longlong)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.longlong)
        self.sos_word_np = np.array([self.e2w[etype]['%s <SOS>' % etype] for etype in self.classes], dtype=np.longlong)
        self.eos_word_np = np.array([self.e2w[etype]['%s <EOS>' % etype] for etype in self.classes], dtype=np.longlong)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        # self.linear_emb = []
        for i, key in enumerate(self.classes):  # 将每个特征都Embedding到256维，Embedding参数是可学习的
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
            # self.linear_emb.append(nn.Linear(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)
        # self.linear_emb = nn.ModuleList(self.linear_emb)

        # linear layer to merge embeddings from different token types
        self.in_linear = nn.Linear(int(np.sum(self.emb_sizes)), bertConfig.d_model)

        self.attention_linear = nn.Sequential(
            nn.Linear(int(np.sum(self.emb_sizes)), np.sum(self.emb_sizes) //2),
            nn.ReLU(),
            nn.Linear(np.sum(self.emb_sizes) // 2, np.sum(self.emb_sizes) // 2),
            nn.ReLU(),
            nn.Linear(np.sum(self.emb_sizes) // 2, int(np.sum(self.emb_sizes))),
            nn.Sigmoid(),
        )


    def forward(self, input_ids, attn_mask=None, output_hidden_states=True, x=None):
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i, key in enumerate(self.classes):
            # if x is None:
            #     embs.append(self.word_emb[i](input_ids[..., i]))
            # else:
            #     emb_result = self.word_emb[i](input_ids[..., i])
            #     linear_result = self.linear_emb[i](x[i])
            #     embs.append(emb_result+(linear_result-linear_result.detach()))
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)

        # embs = self.tw_attention(embs)

        emb_linear = self.in_linear(embs)

        # feed to bert
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y

    def get_rand_tok(self):
        rand=[0]*8
        for i in range(8):
            rand[i]=random.choice(range(self.n_tokens[i]))
        return np.array(rand)

    def tw_attention(self,x):
        weight = self.attention_linear(x)
        return x * weight





# class MidiBert(nn.Module):
#     def __init__(self, bertConfig, e2w, w2e):
#         super().__init__()
#
#         self.bert = BertModel(bertConfig)
#         bertConfig.d_model = bertConfig.hidden_size
#         self.hidden_size = bertConfig.hidden_size
#         self.bertConfig = bertConfig
#
#         self.n_tokens = []
#         self.classes = ['Bar', 'Position', 'Instrument', 'Pitch', 'Duration', 'Velocity', 'TimeSig', 'Tempo']
#         for key in self.classes:
#             self.n_tokens.append(len(e2w[key]))
#         self.emb_sizes = [256] * 8
#         self.e2w = e2w
#         self.w2e = w2e
#
#         # for deciding whether the current input_ids is a <PAD> token
#         self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
#         self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.longlong)
#         self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.longlong)
#         self.sos_word_np = np.array([self.e2w[etype]['%s <SOS>' % etype] for etype in self.classes], dtype=np.longlong)
#         self.eos_word_np = np.array([self.e2w[etype]['%s <EOS>' % etype] for etype in self.classes], dtype=np.longlong)
#
#         # word_emb: embeddings to change token ids into embeddings
#         self.word_emb = []
#         for i, key in enumerate(self.classes):  # 将每个特征都Embedding到256维，Embedding参数是可学习的
#             self.word_emb.append(nn.Linear(self.n_tokens[i], self.emb_sizes[i]))
#         self.word_emb = nn.ModuleList(self.word_emb)
#
#         # linear layer to merge embeddings from different token types
#         self.in_linear = nn.Linear(int(np.sum(self.emb_sizes)), bertConfig.d_model)
#
#         self.attention_linear = nn.Sequential(
#             nn.Linear(int(np.sum(self.emb_sizes)), np.sum(self.emb_sizes) //2),
#             nn.ReLU(),
#             nn.Linear(np.sum(self.emb_sizes) // 2, np.sum(self.emb_sizes) // 2),
#             nn.ReLU(),
#             nn.Linear(np.sum(self.emb_sizes) // 2, int(np.sum(self.emb_sizes))),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, input_ids, attn_mask=None, output_hidden_states=True,x=None):
#         # convert input_ids into embeddings and merge them through linear layer
#         embs = []
#         for i, key in enumerate(self.classes):
#             input = F.one_hot(input_ids[..., i].long(),num_classes=self.n_tokens[i])
#             input = input.float()
#             if x is not None:
#                 input = input.detach() + (x[i]-x[i].detach())
#             embs.append(self.word_emb[i](input))
#         embs = torch.cat([*embs], dim=-1)
#
#         embs = self.tw_attention(embs)
#
#         emb_linear = self.in_linear(embs)
#
#         # feed to bert
#         y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
#         # y = y.last_hidden_state         # (batch_size, seq_len, 768)
#         return y
#
#     def get_rand_tok(self):
#         rand=[0]*8
#         for i in range(8):
#             rand[i]=random.choice(range(self.n_tokens[i]))
#         return np.array(rand)
#
#     def tw_attention(self,x):
#         weight = self.attention_linear(x)
#         return x * weight

class MidiBertLM(nn.Module):
    def __init__(self, midibert: MidiBert):
        super().__init__()

        self.midibert = midibert
        self.mask_lm = MLM(self.midibert.e2w, self.midibert.n_tokens, self.midibert.hidden_size)

    def forward(self, x, attn):
        x = self.midibert(x, attn)
        return self.mask_lm(x)


class MLM(nn.Module):
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)  # 必须用这种方法才能像列表一样访问网络的每层

        self.e2w = e2w

    def forward(self, y):
        # feed to bert
        y = y.hidden_states[-1]

        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))  # (batch_size, seq_len, dict_size)
        return ys

class Masker(nn.Module):
    def __init__(self, midibert, hs):
        super().__init__()
        self.midibert = midibert
        self.linear = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, y, attn, layer=-1):
        # feed to bert
        y = self.midibert(y, attn, output_hidden_states=True)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[layer]
        y = self.linear(y)
        return y.squeeze()

# GRL
# 梯度反转层，这一层正向表现为恒等变换，反向传播是改变梯度的符号，alpha用来平衡域损失的权重。
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices):
        ctx.save_for_backward(x, indices)
        return x.gather(1, indices)

    @staticmethod
    def backward(ctx, grad_output):
        x, indices = ctx.saved_tensors
        grad_x = torch.zeros_like(x).scatter_(1, indices, grad_output)
        return grad_x, None

# class Discriminator(nn.Module):
#     def __init__(self, midibert, hs):
#         super().__init__()
#         self.midibert = midibert
#         self.linear = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(hs, 256),
#             nn.ReLU(),
#             nn.Linear(256, 2),
#         )
#         self.GRL = GRL()
#
#     def forward(self, y, attn, alpha=1, layer=-1, x=None):
#         # feed to bert
#         y = self.midibert(y, attn, output_hidden_states=True, x=x)
#         # y = y.last_hidden_state         # (batch_size, seq_len, 768)
#         y = y.hidden_states[layer]
#         y = self.GRL.apply(y, alpha)
#         y = self.linear(y)
#         return y.squeeze()

class Discriminator(nn.Module):
    def __init__(self, midibert, hs, da=128, r=4):
        super().__init__()
        self.midibert = midibert
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs * r, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.GRL = GRL()

    def forward(self, y, attn, alpha=1, layer=-1, x=None):  # x: (batch, 512, 4)
        y = self.midibert(y, attn, output_hidden_states=True, x=x)  # (batch, 512, 768)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[layer]
        y = self.GRL.apply(y, alpha)
        attn_mat = self.attention(y)  # attn_mat: (batch, r, 512)
        m = torch.bmm(attn_mat, y)  # m: (batch, r, 768)
        flatten = m.view(m.size()[0], -1)  # flatten: (batch, r*768)
        res = self.classifier(flatten)  # res: (batch, class_num)
        return res

class TokenClassification(nn.Module):
    def __init__(self, midibert, class_num, hs):
        super().__init__()

        self.midibert = midibert
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, y, attn, layer=-1):
        # feed to bert
        y = self.midibert(y, attn, output_hidden_states=True)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[layer]
        return self.classifier(y)

class SequenceClassification(nn.Module):
    def __init__(self, midibert, class_num, hs, da=128, r=4):
        super(SequenceClassification, self).__init__()
        self.midibert = midibert
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs * r, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, x, attn, layer=-1):  # x: (batch, 512, 4)
        x = self.midibert(x, attn, output_hidden_states=True)  # (batch, 512, 768)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        x = x.hidden_states[layer]
        attn_mat = self.attention(x)  # attn_mat: (batch, r, 512)
        m = torch.bmm(attn_mat, x)  # m: (batch, r, 768)
        flatten = m.view(m.size()[0], -1)  # flatten: (batch, r*768)
        res = self.classifier(flatten)  # res: (batch, class_num)
        return res

class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        '''
        Args:
            input_dim (int): batch, seq, input_dim
            da (int): number of features in hidden layer from self-attn
            r (int): number of aspects of self-attn
        '''
        super(SelfAttention, self).__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat

