from typing import Any, Dict, List, Mapping, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.modules.tokenizers import Tokenizer
import json
from tqdm import tqdm

class DiscordDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        source: str,
        max_seq_len: int,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        with open("./backrooms.json", "r") as f:
            data = json.load(f)["rooms"]
        self.max_seq_len = max_seq_len
        self._data = []

        for room in data:
            tokenized_messages = []
            for message in tqdm(room):
                tokens = []
                # Start with a special token to indicate the start of a new message
                tokens += [self._tokenizer._encode_special_token("<|start_header_id|>")]
                # Message ID
                tokens += self._tokenizer.encode(message['id'], add_bos=False, add_eos=False)
                # Separate with another special token
                tokens += [self._tokenizer._encode_special_token("<|fim_prefix|>")]
                # Message author
                tokens += self._tokenizer.encode(message["author"], add_bos=False, add_eos=False)
                # Separate with another special token
                tokens += [self._tokenizer._encode_special_token("<|fim_middle|>")]
                # Reply ID
                if message["reference"] is not None:
                    tokens += self._tokenizer.encode(message["reference"], add_bos=False, add_eos=False)
                # Separate with another special token
                tokens += [self._tokenizer._encode_special_token("<|end_header_id|>")]
                # Begin message with new line
                tokens += self._tokenizer.encode("\n", add_bos=False, add_eos=False)
                # Message content
                tokens += self._tokenizer.encode(message["content"], add_bos=False, add_eos=False)
                # End message with special token and double new line
                tokens += [self._tokenizer._encode_special_token("<|eom_id|>")]
                tokens += self._tokenizer.encode("\n\n", add_bos=False, add_eos=False)
                tokenized_messages.append(tokens)
            current_chunk = [self._tokenizer._encode_special_token("<|begin_of_text|>")]
            i = 0
            while True:
                if i == len(tokenized_messages) or len(current_chunk) + len(tokenized_messages[i]) > self.max_seq_len - 1:
                    current_chunk += [self._tokenizer._encode_special_token("<|end_of_text|>")]
                    self._data.append(current_chunk)
                    current_chunk = [self._tokenizer._encode_special_token("<|begin_of_text|>")]
                    if i == len(tokenized_messages):
                        break
                    # Create overlap between chunks
                    i -= 4
                else:
                    current_chunk += tokenized_messages[i]
                    i += 1
        print(f"Number of chunks: {len(self._data)}")


    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        labels = sample.copy()

        return { "tokens": sample, "labels": labels }

def discord_dataset(
    tokenizer: Tokenizer,
    source: str = "liweili/c4_200m",
):
    return DiscordDataset(
        tokenizer=tokenizer,
        source=source,
        max_seq_len=4096,
    )
