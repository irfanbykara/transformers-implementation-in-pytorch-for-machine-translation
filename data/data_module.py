import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class Multi30kDataModule:
    def __init__(self, source_language="en", target_language="de", batch_size=32):
        self.SOURCE_LANGUAGE = source_language
        self.TARGET_LANGUAGE = target_language
        self.BATCH_SIZE = batch_size

        # Special tokens
        self.BOS_TOKEN = "<bos>"
        self.EOS_TOKEN = "<eos>"
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"

        self.BOS_IDX = 0
        self.EOS_IDX = 1
        self.PAD_IDX = 2
        self.UNK_IDX = 3

        self.token_transform = {
            self.SOURCE_LANGUAGE: get_tokenizer("spacy", language="en_core_web_sm"),
            self.TARGET_LANGUAGE: get_tokenizer("spacy", language="de_core_news_sm")
        }

        self.vocab_transform = {}
        self._build_vocabs()

    def _yield_tokens(self, data_iter, language):
        lang_index = 0 if language == self.SOURCE_LANGUAGE else 1
        for pair in data_iter:
            yield self.token_transform[language](pair[lang_index])

    def _build_vocabs(self):
        train_iter = Multi30k(split='train')

        for lang in [self.SOURCE_LANGUAGE, self.TARGET_LANGUAGE]:
            vocab = build_vocab_from_iterator(
                self._yield_tokens(train_iter, lang),
                specials=[self.BOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN],
                special_first=True
            )
            vocab.set_default_index(self.UNK_IDX)
            self.vocab_transform[lang] = vocab

    def _tensor_transform(self, token_ids):
        return torch.cat([
            torch.tensor([self.BOS_IDX]),
            torch.tensor(token_ids),
            torch.tensor([self.EOS_IDX])
        ])

    def _collate_batch(self, batch):
        source_batch, target_batch = [], []

        for source_sample, target_sample in batch:
            source_tokens = self.token_transform[self.SOURCE_LANGUAGE](source_sample)
            target_tokens = self.token_transform[self.TARGET_LANGUAGE](target_sample)

            source_ids = [self.vocab_transform[self.SOURCE_LANGUAGE][tok] for tok in source_tokens]
            target_ids = [self.vocab_transform[self.TARGET_LANGUAGE][tok] for tok in target_tokens]

            source_batch.append(self._tensor_transform(source_ids))
            target_batch.append(self._tensor_transform(target_ids))

        source_batch = pad_sequence(source_batch, padding_value=self.PAD_IDX)
        target_batch = pad_sequence(target_batch, padding_value=self.PAD_IDX)

        return source_batch.T, target_batch.T  # Shape: (B, T)

    def get_dataloader(self, split="train"):
        dataset = list(Multi30k(split=split))
        return DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=(split == "train"), collate_fn=self._collate_batch)
