# src/model/modular.py
from typing import Optional, Dict, Any
import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModel
from torchcrf import CRF
from src.utils import LABELS  # Импортируем LABELS для конфига

class CRFLayer(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, emissions: torch.Tensor, mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()

        if labels is not None:
            # negative log-likelihood (scalar)
            nll = -self.crf(emissions, labels, mask=mask)
            return nll / labels.size(0)
        else:
            return self.crf.decode(emissions, mask=mask)


class ModularTokenClassifier(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "backbone"
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"cls"]

    def __init__(self,
                 config: AutoConfig,
                 base_model_name_or_path: Optional[str] = None,
                 num_labels: int = 9,
                 use_crf: bool = True,
                 extra_dropout: Optional[float] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 ce_weight: float = 0.7):
        super().__init__(config)
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.ce_weight = ce_weight  # weight for CE term when using CRF
        self.class_weights = class_weights  # torch tensor or None

        if base_model_name_or_path:
            self.backbone = AutoModel.from_pretrained(base_model_name_or_path, config=config)
        else:
            self.backbone = AutoModel.from_config(config)

        self.dropout = nn.Dropout(extra_dropout if extra_dropout is not None else getattr(config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRFLayer(num_labels) if use_crf else None

        self.post_init()

    @classmethod
    def from_pretrained_base(cls, pretrained_path: str, use_crf: bool = True, num_labels: int = 9, extra_dropout: Optional[float] = None, class_weights: Optional[torch.Tensor] = None, ce_weight: float = 0.7):
        config = AutoConfig.from_pretrained(pretrained_path)
        config.num_labels = num_labels

        # Numeric id2label / label2id required by HF
        config.id2label = {i: label for i, label in enumerate(LABELS)}
        config.label2id = {label: i for i, label in enumerate(LABELS)}

        return cls(config, base_model_name_or_path=pretrained_path, num_labels=num_labels, use_crf=use_crf, extra_dropout=extra_dropout, class_weights=class_weights, ce_weight=ce_weight)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        import inspect

        allowed_args = inspect.signature(self.backbone.forward).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_args}

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **filtered_kwargs)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            # TRAINING
            if self.use_crf and self.crf is not None:
                # safe labels: clamp and mask pad tokens to 0 (or other)
                safe_labels = labels.clone()
                if mask is not None:
                    safe_labels[~mask] = 0
                safe_labels = torch.clamp(safe_labels, 0, self.num_labels - 1)

                crf_loss = self.crf(emissions=emissions, mask=mask, labels=safe_labels)

                # additionally compute weighted CE on active positions to nudge emissions for rare classes
                if self.class_weights is not None:
                    device = emissions.device
                    weights = self.class_weights.to(device)
                    loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

                if mask is not None:
                    active_loss = mask.view(-1) == 1
                    active_logits = emissions.view(-1, self.num_labels)
                    active_labels = labels.view(-1)
                    ce_loss = loss_fct(active_logits[active_loss], active_labels[active_loss])
                else:
                    ce_loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))

                total_loss = crf_loss + self.ce_weight * ce_loss
                return {"loss": total_loss, "logits": emissions}

            else:
                # plain CrossEntropy
                if self.class_weights is not None:
                    device = emissions.device
                    weights = self.class_weights.to(device)
                    loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

                if mask is not None:
                    active_loss = mask.view(-1) == 1
                    active_logits = emissions.view(-1, self.num_labels)
                    active_labels = labels.view(-1)
                    loss = loss_fct(active_logits[active_loss], active_labels[active_loss])
                else:
                    loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
                return {"loss": loss, "logits": emissions}
        else:
            # INFERENCE
            if self.use_crf and self.crf is not None:
                preds = self.crf(emissions=emissions, mask=mask)
                return {"logits": emissions, "predictions": preds}
            else:
                probs = torch.softmax(emissions, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                return {"logits": emissions, "predictions": preds}
    
    def freeze_base(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_base(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def save_run(self, run_dir: str, tokenizer=None, meta: Optional[dict] = None) -> None:
        os.makedirs(run_dir, exist_ok=True)
        #  сохраняем с safe_serialization=False
        self.save_pretrained(run_dir, safe_serialization=False)
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))
        from src.utils import save_model_parts
        save_model_parts(self, run_dir, save_full=True, tokenizer=tokenizer, meta=meta)