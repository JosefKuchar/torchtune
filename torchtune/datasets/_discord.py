from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.config._utils import _get_chat_format
from torchtune.data import (
    ChatFormat,
    CROSS_ENTROPY_IGNORE_IDX,
    Message,
    openai_to_llama2_messages,
    sharegpt_to_llama2_messages,
    validate_messages,
)
from torchtune.modules.tokenizers import Tokenizer


class DiscordDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        source: str,
        max_seq_len: int,
        train_on_input: bool = True,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(
            "json", data_files="./backrooms.json", field="rooms", split="train", **load_dataset_kwargs
        )
        print(len(self._data))
        exit()
        self.max_seq_len = max_seq_len
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        # messages = self._convert_to_messages(sample, self.train_on_input)
        # if self.chat_format is not None:
        #     messages = self.chat_format.format(messages)
        # validate_messages(messages)
        # tokens, mask = self._tokenizer.tokenize_messages(
        #     messages, max_seq_len=self.max_seq_len
        # )
        # # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        # labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        # assert len(tokens) == len(labels)

        return [], []

def discord_dataset(
    tokenizer: Tokenizer,
    source: str = "liweili/c4_200m",
    train_on_input: bool = False
):
    return DiscordDataset(
        tokenizer=tokenizer,
        source=source,
        train_on_input=train_on_input,
    )
