"""
Data collator for NutriPlan multi-task learning
Handles variable-length sequences with padding
"""

import torch
from typing import Dict, List, Any


class NutriPlanCollator:
    """Custom collator for NutriPlan that handles padding"""

    def __init__(self, tokenizer, padding=True, max_length=None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            padding: Whether to pad sequences
            max_length: Maximum sequence length (None = pad to longest in batch)
        """
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with padding

        Args:
            batch: List of samples from dataset

        Returns:
            Padded batch as dict of tensors
        """
        # Separate input_ids, attention_mask, labels
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item.get('attention_mask') for item in batch]
        labels = [item.get('labels') for item in batch]

        # Pad sequences
        if self.padding:
            # Find max length in batch
            max_len = max(len(ids) for ids in input_ids)
            if self.max_length:
                max_len = min(max_len, self.max_length)

            # Pad input_ids
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []

            for i in range(len(batch)):
                ids = input_ids[i]
                mask = attention_mask[i] if attention_mask[i] is not None else torch.ones_like(ids)
                labs = labels[i] if labels[i] is not None else ids.clone()

                # Truncate if needed
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    mask = mask[:max_len]
                    labs = labs[:max_len]

                # Pad if needed
                padding_length = max_len - len(ids)
                if padding_length > 0:
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    ids = torch.cat([ids, torch.full((padding_length,), pad_id, dtype=ids.dtype)])
                    mask = torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])
                    labs = torch.cat([labs, torch.full((padding_length,), -100, dtype=labs.dtype)])  # -100 ignored in loss

                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
                padded_labels.append(labs)

            # Stack into tensors
            return {
                'input_ids': torch.stack(padded_input_ids),
                'attention_mask': torch.stack(padded_attention_mask),
                'labels': torch.stack(padded_labels)
            }
        else:
            # No padding (not recommended)
            return {
                'input_ids': torch.stack(input_ids),
                'attention_mask': torch.stack(attention_mask) if attention_mask[0] is not None else None,
                'labels': torch.stack(labels) if labels[0] is not None else None
            }
