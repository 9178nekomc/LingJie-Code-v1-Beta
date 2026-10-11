# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  多模块 Head
num_classes 全部参数化（代码模型与未来的 Sentence 模型共享 JAM 底座，
推理管线的"翻译层"可插拔：Code 走沙箱，Sentence 走合理性判断）。

5 个 Head 构成串行级联决策链（推理管线中硬编码决策逻辑）：
  IntentHead   (2)   : 一票否决，置信度 < 0.75 直接 Cannot
  ParserHead   (4)   : 逐 Token 标注（动作词/数据字面量/变量名/噪声）
  VerbHead     (20)  : 动作路由，20 个具体动作
  TemplateHead (100) : 骨架选择
  CriticHead   (2)   : 一票否决 + 触发修复循环

池化策略：取每个序列 <resp> 决策点的隐状态（pool_idx 显式传入），
<resp> 之前只有 prompt，因果表征与增量推理时完全一致。
"""
import torch
import torch.nn as nn


class SequencePool(nn.Module):
    """按 pool_idx [b] 收集各样本决策点向量"""

    def __init__(self, pad_token_id: int, resp_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.resp_token_id = resp_token_id

    @staticmethod
    def find_pool_idx(input_ids: torch.Tensor, resp_token_id: int,
                      pad_token_id: int) -> torch.Tensor:
        """优先取第一个 <resp>；若无则取最后一个非 pad token。返回 [b]"""
        b, t = input_ids.shape
        resp_pos = (input_ids == resp_token_id).float().argmax(dim=1)   # 无则 0
        has_resp = (input_ids == resp_token_id).any(dim=1)
        valid = (input_ids != pad_token_id)
        last_valid = (valid.float() * torch.arange(t, device=input_ids.device)
                      .float().unsqueeze(0)).argmax(dim=1)
        idx = torch.where(has_resp, resp_pos, last_valid)
        return idx.long()

    def forward(self, hidden: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
        idx = pool_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)                   # [b, d]


class IntentHead(nn.Module):
    """2 分类：0 = 代码请求(放行) / 1 = 非代码请求(Cannot)。一票否决权。"""

    def __init__(self, d_model: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class ParserHead(nn.Module):
    """4 分类，逐 Token：0=动作词 1=数据字面量 2=变量名 3=噪声"""

    def __init__(self, d_model: int, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class VerbHead(nn.Module):
    """20 分类动作路由"""

    ACTIONS = ["print", "if", "for", "while", "assign", "input", "len",
               "append", "pop", "sort", "max", "min", "sum", "def", "return",
               "import", "open", "read", "write", "CANNOT"]
    ACTION2ID = {a: i for i, a in enumerate(ACTIONS)}

    def __init__(self, d_model: int, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class TemplateHead(nn.Module):
    """100 分类骨架选择"""

    def __init__(self, d_model: int, num_classes: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class CriticHead(nn.Module):
    """2 分类：1 = 语义一致 / 0 = 不一致。一票否决 + 触发修复循环。"""

    def __init__(self, d_model: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class MultiHeads(nn.Module):
    """5 Head 组合，池化点为各样本 <resp> 决策位置"""

    def __init__(self, d_model: int, pad_token_id: int, resp_token_id: int,
                 intent_classes: int = 2, parser_classes: int = 4,
                 verb_classes: int = 20, template_classes: int = 100,
                 critic_classes: int = 2):
        super().__init__()
        self.pool = SequencePool(pad_token_id, resp_token_id)
        self.intent = IntentHead(d_model, intent_classes)
        self.parser = ParserHead(d_model, parser_classes)
        self.verb = VerbHead(d_model, verb_classes)
        self.template = TemplateHead(d_model, template_classes)
        self.critic = CriticHead(d_model, critic_classes)

    def forward(self, hidden: torch.Tensor, pool_idx: torch.Tensor) -> dict:
        pooled = self.pool(hidden, pool_idx)
        return {
            "intent": self.intent(pooled),        # [b, 2]
            "parser": self.parser(hidden),        # [b, t, 4] 逐 token
            "verb": self.verb(pooled),            # [b, 20]
            "template": self.template(pooled),    # [b, 100]
            "critic": self.critic(pooled),        # [b, 2]
        }
