#!/usr/bin/env python3

# This file is part of CorPipe <https://github.com/ufal/crac2026-corpipe>.
#
# Copyright 2026 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import datetime
import json
import os
import pickle
import shutil
import re
from typing import Callable

import huggingface_hub
import minnt
import numpy as np
import torch
import transformers
import udapi
import udapi.block.corefud.movehead
import udapi.block.corefud.removemisc

minnt.require_version("1.1")

parser = argparse.ArgumentParser()
parser.add_argument("--adafactor", default=False, action="store_true", help="Use Adafactor.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--compile", default=False, action="store_true", help="Compile the model.")
parser.add_argument("--depth", default=5, type=int, help="Constrained decoding depth.")
parser.add_argument("--dev", default=None, nargs="*", type=str, help="Predict dev (treebanks).")
parser.add_argument("--encoder", default="google/mt5-large", type=str, help="MLM encoder model.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--exp", default="", type=str, help="Exp name.")
parser.add_argument("--label_smoothing", default=0.2, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=5e-4, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default=False, action="store_true", help="Decay LR.")
parser.add_argument("--load", default=[], type=str, nargs="*", help="Models to load.")
parser.add_argument("--right", default=50, type=int, help="Reserved space for right context, if any.")
parser.add_argument("--sampling_exponent", default=0.5, type=float, help="Sampling exponent during training.")
parser.add_argument("--sampling_mode", default="sentences", choices=["sentences", "words"], help="Sampling mode during training.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--segment", default=512, type=int, help="Segment size")
parser.add_argument("--steps_per_epoch", default=10_000, type=int, help="Update steps (batches) per epoch.")
parser.add_argument("--test", default=None, nargs="*", type=str, help="Predict test (treebanks).")
parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train", default=False, action="store_true", help="Perform training.")
parser.add_argument("--treebanks", default=[], nargs="+", type=str, help="Data.")
parser.add_argument("--warmup", default=0.1, type=float, help="Warmup ratio.")
parser.add_argument("--zeros_per_parent", default=2, type=int, help="Zeros per parent.")


class Dataset:
    TOKEN_EMPTY = "[TOKEN_EMPTY]"
    TOKEN_CLS = "[TOKEN_CLS]"
    ZDEPREL_NONE = 0

    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizerFast) -> None:
        self._cls = tokenizer.cls_token_id
        self._sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self._path = path
        if self._cls is None:
            self._cls = tokenizer.vocab[self.TOKEN_CLS]

        # Create the tokenized documents if they do not exist
        cache_path = f"{path}.mentions1.{os.path.basename(tokenizer.name_or_path)}"
        if not os.path.exists(cache_path) or os.path.getmtime(cache_path) <= os.path.getmtime(path):
            # Parse with Udapi
            if not os.path.exists(f"{path}.mentions1") or os.path.getmtime(f"{path}.mentions1") <= os.path.getmtime(path):
                docs, new_doc = [], []
                for doc in udapi.block.read.conllu.Conllu(files=[path]).read_documents():
                    for tree in doc.trees:
                        if tree.newdoc is not None and new_doc:
                            docs.append(new_doc)
                            new_doc = []
                        words, coref_mentions = [], set()
                        for node in tree.descendants:
                            words.append(node.form)
                            coref_mentions.update(node.coref_mentions)
                        for enode in tree.empty_nodes:
                            coref_mentions.update(enode.coref_mentions)

                        dense_mentions = []
                        for mention in [mention for mention in coref_mentions if not mention.head.is_empty()]:
                            span = [word for word in mention.words if not word.is_empty()]
                            start = end = span.index(mention.head)
                            while start > 0 and span[start - 1].ord + 1 == span[start].ord:
                                start -= 1
                            while end < len(span) - 1 and span[end].ord + 1 == span[end + 1].ord:
                                end += 1
                            dense_mentions.append(((span[start].ord - 1, span[end].ord - 1), mention.entity.eid, start > 0 or end + 1 < len(span)))
                        dense_mentions = sorted(dense_mentions, key=lambda x: (x[0][0], -x[0][1], x[2]))

                        mentions = []
                        for i, mention in enumerate(dense_mentions):
                            if i and dense_mentions[i - 1][0] == mention[0]:
                                print(f"Multiple same mentions {mention[2]}/{dense_mentions[i-1][2]} in sent_id {tree.sent_id}: {tree.get_sentence()}", flush=True)
                                continue
                            mentions.append((mention[0][0], mention[0][1], mention[1]))

                        zero_mentions = []
                        for mention in [mention for mention in coref_mentions if mention.head.is_empty()]:
                            if len(mention.words) > 1:
                                print(f"A empty-node-head mention with multiple words {mention.words} in sent_id {tree.sent_id}: {tree.get_sentence()}", flush=True)
                            assert len(mention.head.deps) >= 1
                            zero_mentions.append((mention.head.deps[0]["parent"].ord - 1, mention.head.deps[0]["deprel"], mention.entity.eid))
                        zero_mentions = sorted(zero_mentions)
                        new_doc.append((words, mentions, zero_mentions))
                if new_doc:
                    docs.append(new_doc)
                with open(f"{path}.mentions1", "wb") as cache_file:
                    pickle.dump(docs, cache_file, protocol=3)
            with open(f"{path}.mentions1", "rb") as cache_file:
                docs = pickle.load(cache_file)

            # Tokenize the data, generate stack operations and subword mentions
            self.docs = []
            for doc in docs:
                new_doc = []
                for words, mentions, zero_mentions in doc:
                    subwords, word_indices, word_tags, subword_mentions, stack = [], [], [], [], []
                    for i in range(len(words)):
                        word_indices.append(len(subwords))
                        word = (" " if "robeczech" in tokenizer.name_or_path or "t5gemma" in tokenizer.name_or_path else "") + words[i]
                        subword = tokenizer.encode(word, add_special_tokens=False)
                        assert len(subword) > 0
                        if subword[0] == 6 and "xlm-r" in tokenizer.name_or_path:  # Hack: remove the space-only token in XLM-R
                            subword = subword[1:]
                        assert len(subword) > 0
                        subwords.extend(subword)

                        tag = []
                        for _ in range(2):
                            for j in reversed(range(len(stack))):
                                start, end, eid = stack[j]
                                if end == i:
                                    tag.append(f"POP:{len(stack)-j}")
                                    subword_mentions.append((start, word_indices[-1], eid))
                                    stack.pop(j)
                            while mentions and mentions[0][0] == i:
                                tag.append("PUSH")
                                stack.append((word_indices[-1], mentions[0][1], mentions[0][2]))
                                mentions = mentions[1:]
                        word_tags.append(",".join(tag))
                    assert len(stack) == 0

                    word_zdeprels = [[] for _ in range(len(words))]
                    for parent, deprel, eid in zero_mentions:
                        word_zdeprels[parent].append(deprel)
                        subword_mentions.append((word_indices[parent], -len(word_zdeprels[parent]), eid))
                    subword_mentions = sorted(subword_mentions, key=lambda x: (x[0], -x[1]))

                    new_doc.append((subwords, word_indices, word_tags, word_zdeprels, subword_mentions))
                self.docs.append(new_doc)

            with open(cache_path, "wb") as cache_file:
                pickle.dump(self.docs, cache_file, protocol=3)
        with open(cache_path, "rb") as cache_file:
            self.docs = pickle.load(cache_file)

    @staticmethod
    def create_tags(trains: list["Dataset"]) -> list[str]:
        tags = set()
        for train in trains:
            for doc in train.docs:
                for _, _, word_tags, _, _ in doc:
                    tags.update(word_tags)
        return sorted(tags)

    @staticmethod
    def create_zdeprels(trains: list["Dataset"]) -> list[str]:
        zdeprels = set()
        for train in trains:
            for doc in train.docs:
                for _, _, _, word_zdeprels, _ in doc:
                    zdeprels.update(zdeprel for zdeprels in word_zdeprels for zdeprel in zdeprels)
        return ["[NONE]"] + sorted(zdeprels)  # prepend ZDEPREL_NONE

    @staticmethod
    def allowed_tag_transitions(tags: list[str], depth: int) -> torch.Tensor:
        tags = [f"{d}{',' if tag else ''}{tag}" for d in range(depth) for tag in tags]
        allowed = torch.empty(len(tags), len(tags), dtype=torch.float32)
        for i, tag_i in enumerate(tags):
            for j, tag_j in enumerate(tags):
                i_parts = tag_i.split(",")
                i_depth = int(i_parts[0])
                j_depth = int(tag_j.split(",")[0])
                for command in i_parts[1:]:
                    i_depth += 1 if command == "PUSH" and i_depth >= 0 else -1
                allowed[i, j] = 0 if i_depth == j_depth else -torch.inf
        return allowed

    def dataset(self, tags_map: dict[str, int], zdeprels_map: dict[str, int], train: bool, args: argparse.Namespace) -> list:
        segment_size = args.segment
        if "proiel" in self._path:  # hard-code maximum segment size for Proiel to 512
            segment_size = min(512, args.segment)
        dataset = []
        for doc in self.docs:
            p_subwords, p_subword_mentions = [], []
            for doc_i, (subwords, word_indices, word_tags, word_zdeprels, subword_mentions) in enumerate(doc):
                subword_mentions = [(s, e, eid) for s, e, eid in subword_mentions if e >= -args.zeros_per_parent]
                if not train and len(subwords) + 4 > segment_size:
                    print("Truncating a long sentence during prediction")
                    subwords = subwords[:segment_size - 4]
                assert train or len(subwords) + 4 <= segment_size
                if len(subwords) + 4 <= segment_size:
                    right_reserve = min((segment_size - 4 - len(subwords)) // 2, args.right or 0)
                    context = min(segment_size - 4 - len(subwords) - right_reserve, len(p_subwords))
                    word_indices = [context + 2 + i for i in word_indices + [len(subwords)]]
                    e_subwords = [self._cls, *p_subwords[len(p_subwords) - context:], self._sep, *subwords, self._sep]
                    if args.right is not None:
                        i = doc_i + 1
                        while i < len(doc) and len(e_subwords) + 1 < segment_size:
                            e_subwords.extend(doc[i][0][:segment_size - len(e_subwords) - 1])
                            i += 1
                    e_subwords.append(self._sep)

                    output = (torch.tensor(e_subwords), torch.tensor(word_indices))
                    if train:
                        offset = len(p_subwords) - context
                        prev = [(s - offset + 1, e if e < 0 else e - offset + 1, eid) for s, e, eid in p_subword_mentions if s >= offset]
                        prev_pos = np.array([[s, e] for s, e, _ in prev], dtype=np.int64).reshape([-1, 2])
                        prev_eid = np.array([eid for _, _, eid in prev], dtype=str)
                        curr = [(context + 2 + s, e if e < 0 else context + 2 + e, eid) for s, e, eid in subword_mentions]
                        curr_pos = np.array([[s, e] for s, e, _ in curr], dtype=np.int64).reshape([-1, 2])
                        curr_eid = np.array([eid for _, _, eid in curr], dtype=str)
                        mask = curr_pos[:, 0, None] > np.concatenate([prev_pos[:, 0], curr_pos[:, 0]])[None, :]
                        diag = np.pad(np.eye(len(curr_pos), dtype=np.bool), [[0, 0], [len(prev_pos), 0]])
                        gold = (curr_eid[:, None] == np.concatenate([prev_eid, curr_eid])[None, :]) * mask
                        gold = np.where(np.sum(gold, axis=1, keepdims=True) > 0, gold, diag)
                        gold = gold / np.sum(gold, axis=1, keepdims=True, dtype=np.float32)
                        mask = mask | diag
                        if args.label_smoothing:
                            gold = (1 - args.label_smoothing) * gold + args.label_smoothing * (mask / np.sum(mask, axis=1, keepdims=True, dtype=np.float32))
                        gold = np.where(mask, gold, -1)

                        word_tags = [tags_map[tag] for tag in word_tags]
                        word_zdeprels_padded = np.full([len(word_tags), args.zeros_per_parent], -1, np.int64)
                        for zdeprels_padded, zdeprels in zip(word_zdeprels_padded, word_zdeprels):
                            zdeprels_padded[:min(args.zeros_per_parent, len(zdeprels) + 1)] = (
                                [zdeprels_map[zdeprel] for zdeprel in zdeprels] + [self.ZDEPREL_NONE])[:args.zeros_per_parent]

                        output = (output, tuple(map(torch.as_tensor, (word_tags, word_zdeprels_padded, np.concatenate([prev_pos, curr_pos], axis=0), curr_pos, gold))))
                    dataset.append(output)

                p_subword_mentions.extend((s + len(p_subwords), e if e < 0 else e + len(p_subwords), eid) for s, e, eid in subword_mentions)
                p_subwords.extend(subwords)
        return dataset

    @staticmethod
    def padded_batch(train: bool) -> Callable[[list], tuple]:
        def collate(batch: list) -> tuple:
            if train:
                batch, outputs = zip(*batch)
            subwords, word_indices = zip(*batch)
            subwords = torch.nn.utils.rnn.pad_sequence(subwords, batch_first=True, padding_value=-1)
            word_indices = torch.nn.utils.rnn.pad_sequence(word_indices, batch_first=True, padding_value=-1)
            batch = (subwords, word_indices)
            if train:
                word_tags, word_zdeprels, ment_pos, curr_pos, gold = zip(*outputs)
                word_tags = torch.nn.utils.rnn.pad_sequence(word_tags, batch_first=True, padding_value=-1)
                word_zdeprels = torch.nn.utils.rnn.pad_sequence(word_zdeprels, batch_first=True, padding_value=-1)
                ment_pos = torch.nn.utils.rnn.pad_sequence(ment_pos, batch_first=True, padding_value=0)
                curr_pos = torch.nn.utils.rnn.pad_sequence(curr_pos, batch_first=True, padding_value=0)
                gold = torch.stack(
                    [torch.nn.functional.pad(item, (0, ment_pos.shape[1] - item.shape[1] + 1, 0, curr_pos.shape[1] - item.shape[0] + 1), value=-1) for item in gold])[:, :-1, :-1]
                batch = (batch, (word_tags, word_zdeprels, ment_pos, curr_pos, gold))
            return batch
        return collate

    def save_mentions(self, path: str, mentions: list[list[tuple[int, int, int]]], zero_mentions: list[list[tuple[int, str, int]]]) -> None:
        doc = udapi.block.read.conllu.Conllu(files=[self._path]).read_documents()[0]
        udapi.block.corefud.removemisc.RemoveMisc(attrnames="Entity,SplitAnte,Bridge").apply_on_document(doc)

        entities = {}
        for i, tree in enumerate(doc.trees):
            tree.empty_nodes = []  # Drop existing empty nodes
            for node in tree.descendants:  # Remove references to empty nodes also from DEPS, by replacing them by the main dependency edge
                if "." in node.raw_deps:
                    node.raw_deps = f"{node.parent.ord}:{node.deprel}"
            ords = {}
            for parent, deprel, eid in zero_mentions[i]:  # Add predicted empty nodes
                tree.create_empty_child()
                ords[parent] = ords.get(parent, 0) + 1
                tree.empty_nodes[-1].ord = f"{parent+1}.{ords[parent]}"
                tree.empty_nodes[-1].raw_deps = f"{parent+1}:{deprel}"
                if eid not in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention([tree.empty_nodes[-1]], entity=entities[eid])
            nodes = tree.descendants_and_empty
            for start, end, eid in mentions[i]:
                if eid not in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention([node for node in nodes if start <= node.ord - 1 <= end], entity=entities[eid])
        doc._eid_to_entity = {entity._eid: entity for entity in sorted(entities.values())}
        udapi.block.corefud.movehead.MoveHead(bugs='ignore').apply_on_document(doc)
        udapi.block.write.conllu.Conllu(files=[path]).apply_on_document(doc)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[torch.utils.data.Dataset]) -> None:
        self._data = []
        self._ranges = [0]
        for dataset in datasets:
            self._data.extend(dataset)
            self._ranges.append(len(self._data))

    def __len__(self) -> int:
        return self._ranges[-1]

    def __getitem__(self, index: int):
        return self._data[index]

    def sampler(self, args: argparse.Namespace) -> torch.utils.data.Sampler:
        class TrainSampler(torch.utils.data.Sampler):
            def __init__(self, train_dataset) -> None:
                self._data = train_dataset._data
                self._ranges = train_dataset._ranges
                self._examples_per_epoch = args.steps_per_epoch * args.batch_size
                self._generator = torch.Generator().manual_seed(args.seed)

                if args.sampling_mode == "sentences":
                    dataset_weights = np.array([self._ranges[i + 1] - self._ranges[i] for i in range(len(self._ranges) - 1)], np.float32)
                elif args.sampling_mode == "words":
                    dataset_weights = np.array([sum(len(s[0][1]) - 1 for s in self._data[self._ranges[i]:self._ranges[i + 1]]) for i in range(len(self._ranges) - 1)], np.float32)
                else:
                    raise ValueError(f"Unknown sampling mode '{args.sampling_mode}'")
                dataset_weights = dataset_weights ** args.sampling_exponent
                dataset_weights /= np.sum(dataset_weights)
                print(*(f"{100*weight:.1f}" for weight in dataset_weights), flush=True)
                self._dataset_sizes = np.array(dataset_weights * self._examples_per_epoch, np.int32)
                self._dataset_sizes[:self._examples_per_epoch - np.sum(self._dataset_sizes)] += 1
                self._dataset_indices = [[] for _ in self._dataset_sizes]

            def __len__(self) -> int:
                return self._examples_per_epoch

            def __iter__(self) -> iter:
                indices = []
                for i in range(len(self._dataset_sizes)):
                    required = self._dataset_sizes[i]
                    while required:
                        if not len(self._dataset_indices[i]):
                            self._dataset_indices[i] = self._ranges[i] + torch.randperm(
                                self._ranges[i + 1] - self._ranges[i], generator=self._generator)
                        indices.append(self._dataset_indices[i][:required])
                        self._dataset_indices[i] = self._dataset_indices[i][required:]
                        required -= len(indices[-1])
                indices = torch.cat(indices, dim=0)
                return iter(indices[torch.randperm(len(indices), generator=self._generator)].tolist())
        return TrainSampler(self)


class Model(minnt.TrainableModule):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, tags: list[str], zdeprels: list[str], args: argparse.Namespace) -> None:
        super().__init__()
        self._tags = tags
        self._zdeprels = zdeprels
        self._args = args

        assert tags[0] == ""  # Index 0 is used as a boundary condition during decoding
        self.register_buffer("_allowed_tag_transitions", Dataset.allowed_tag_transitions(tags, args.depth), persistent=False)

        config_overrides = {}
        if "umt5" in args.encoder:
            self._encoder = transformers.UMT5EncoderModel
            self._encoder_config = transformers.UMT5Config
        elif "mt5" in args.encoder:
            self._encoder = transformers.MT5EncoderModel
            self._encoder_config = transformers.MT5Config
        elif "t5gemma" in args.encoder:
            self._encoder = transformers.T5GemmaEncoderModel
            self._encoder_config = transformers.T5GemmaConfig
            config_overrides["is_encoder_decoder"] = False
        else:
            self._encoder = transformers.AutoModel
            self._encoder_config = transformers.AutoConfig

        if not args.load:
            self._encoder = self._encoder.from_pretrained(args.encoder, **config_overrides)
        else:
            self._encoder = getattr(self._encoder, "from_config", self._encoder)(self._encoder_config.from_pretrained(args.encoder, **config_overrides))

        if hasattr(self._encoder.config, "hidden_size"):
            encoder_hidden_size = self._encoder.config.hidden_size
        elif hasattr(self._encoder.config, "encoder") and hasattr(self._encoder.config.encoder, "hidden_size"):
            encoder_hidden_size = self._encoder.config.encoder.hidden_size
        else:
            raise ValueError("Cannot determine the encoder hidden size from the model configuration.")

        self._encoder.resize_token_embeddings(len(tokenizer.vocab))
        self._dense_hidden_embeddings = torch.nn.Linear(encoder_hidden_size, 4 * encoder_hidden_size)
        self._dense_hidden_q = torch.nn.Linear(2 * encoder_hidden_size, 4 * encoder_hidden_size)
        self._dense_hidden_k = torch.nn.Linear(2 * encoder_hidden_size, 4 * encoder_hidden_size)
        self._dense_hidden_tags = torch.nn.Linear(encoder_hidden_size, 4 * encoder_hidden_size)
        self._dense_hidden_zdeprels = torch.nn.Linear(encoder_hidden_size, 4 * encoder_hidden_size)
        self._dense_embeddings = torch.nn.Linear(4 * encoder_hidden_size, (1 + args.zeros_per_parent) * encoder_hidden_size)
        self._dense_q = torch.nn.Linear(4 * encoder_hidden_size, encoder_hidden_size, bias=False)
        self._dense_k = torch.nn.Linear(4 * encoder_hidden_size, encoder_hidden_size, bias=False)
        self._dense_tags = torch.nn.Linear(4 * encoder_hidden_size, len(tags))
        self._dense_zdeprels = torch.nn.Linear(4 * encoder_hidden_size, len(zdeprels))

    def configure(self, train: torch.utils.data.DataLoader) -> None:
        args = self._args
        if args.adafactor:
            optimizer = minnt.optimizers.Adafactor(self.parameters(), lr=args.learning_rate, relative_step=False)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        scheduler = minnt.schedulers.GenericDecay(optimizer, args.epochs * len(train), "cosine" if args.learning_rate_decay else "none", warmup=args.warmup)
        super().configure(optimizer=optimizer, scheduler=scheduler, logdir=args.logdir)

    def forward(self, subwords: torch.Tensor, word_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_mask = subwords >= 0
        embeddings = self._encoder(torch.relu(subwords), attention_mask=attention_mask).last_hidden_state
        embeddings = self._dense_embeddings(torch.relu(self._dense_hidden_embeddings(embeddings)))
        embeddings, zero_embeddings = embeddings.tensor_split([embeddings.shape[-1] // (1 + self._args.zeros_per_parent)], dim=-1)

        words = torch.gather(embeddings, 1, torch.relu(word_indices[:, :-1]).unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]))
        tag_logits = self._dense_tags(torch.relu(self._dense_hidden_tags(words)))

        zero_embeddings = zero_embeddings.unflatten(-1, (self._args.zeros_per_parent, -1))
        zdeprel_logits = torch.stack([
            self._dense_zdeprels(torch.relu(self._dense_hidden_zdeprels(
                zero_embeddings[..., i, :].gather(1, torch.relu(word_indices[:, :-1]).unsqueeze(-1).expand(-1, -1, zero_embeddings.shape[-1])))))
            for i in range(self._args.zeros_per_parent)
        ], dim=-2)

        return embeddings, zero_embeddings.flatten(1, 2), tag_logits, zdeprel_logits

    def compute_antecedents(self, embeddings, zero_embeddings, mentions, current) -> torch.Tensor:
        mentions_embedded = torch.gather(
            embeddings[:, :, torch.newaxis, :].expand(-1, -1, 2, -1), 1,
            torch.relu(mentions[:, :, :, torch.newaxis].expand(-1, -1, -1, embeddings.shape[-1])),
        ).flatten(2)
        zero_mentions_embedded = zero_embeddings.gather(
            1, torch.relu(self._args.zeros_per_parent * mentions[..., 0] + -(mentions[..., 1] + 1)).unsqueeze(-1).expand(-1, -1, zero_embeddings.shape[-1])
        ).tile([2])
        mentions_embedded = torch.where(mentions[..., 1:] >= 0, mentions_embedded, zero_mentions_embedded)
        keys = self._dense_k(torch.relu(self._dense_hidden_k(mentions_embedded)))
        current_embedded = torch.gather(
            embeddings[:, :, torch.newaxis, :].expand(-1, -1, 2, -1), 1,
            torch.relu(current[:, :, :, torch.newaxis].expand(-1, -1, -1, embeddings.shape[-1])),
        ).flatten(2)
        zero_current_embedded = zero_embeddings.gather(
            1, torch.relu(self._args.zeros_per_parent * current[..., 0] + -(current[..., 1] + 1)).unsqueeze(-1).expand(-1, -1, zero_embeddings.shape[-1])
        ).tile([2])
        current_embedded = torch.where(current[..., 1:] >= 0, current_embedded, zero_current_embedded)
        queries = self._dense_q(torch.relu(self._dense_hidden_q(current_embedded)))
        weights = (queries @ keys.mT) / (self._dense_q.out_features ** 0.5)
        return weights

    def compute_loss(self, y_pred, y_true, subwords, word_indices) -> torch.Tensor:
        embeddings, zero_embeddings, tag_logits, zdeprel_logits = y_pred
        word_tags, word_zdeprels, all_mentions, current_mentions, gold_mentions = y_true

        # Tagging part
        tag_loss = torch.nn.functional.cross_entropy(tag_logits.movedim(-1, 1), word_tags, ignore_index=-1, label_smoothing=self._args.label_smoothing)
        # Zdeprel part
        zdeprel_loss = torch.nn.functional.cross_entropy(zdeprel_logits.movedim(-1, 1), word_zdeprels, ignore_index=-1, label_smoothing=self._args.label_smoothing)
        # Antecedent part
        antecedent_logits = self.compute_antecedents(embeddings, zero_embeddings, all_mentions, current_mentions)
        gold_mask = gold_mentions >= 0
        antecedent_logits = gold_mask * antecedent_logits + (~gold_mask) * -1e9
        current_mentions_valid = torch.any(gold_mask, dim=-1, keepdim=True)
        antecedent_loss = torch.nn.functional.cross_entropy(
            antecedent_logits.masked_select(current_mentions_valid).view(-1, antecedent_logits.shape[-1]),
            torch.relu(gold_mentions).masked_select(current_mentions_valid).view(-1, gold_mentions.shape[-1]))

        return {"tag_loss": tag_loss, "zdeprel_loss": zdeprel_loss, "antecedent_loss": antecedent_loss}

    def decode_mentions(self, logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # Prepare logits and correct boundary conditions.
        logits = logits.tile([1, 1, self._args.depth])  # duplicate logits according to depth
        logits.masked_fill_((~valid_mask.unsqueeze(-1)) & (torch.arange(logits.shape[-1], device=logits.device) >= 1), -1e9)  # force tag 0 for padding positions
        logits[:, 0, self._allowed_tag_transitions[0, :] == -torch.inf] = -1e9  # the first tag must be such that it can follow tag 0
        logits[:, -1, self._allowed_tag_transitions[:, 0] == -torch.inf] = -1e9  # the last tag must be such that it leads to tag 0

        # Alpha and beta computation.
        alphas = torch.zeros_like(logits)
        betas = torch.zeros_like(alphas, dtype=torch.int64)
        for t in range(logits.shape[1]):
            alphas[:, t] = logits[:, t]
            if t > 0:
                betas[:, t] = torch.argmax(alphas[:, t - 1, :, torch.newaxis] + self._allowed_tag_transitions, dim=1)
                alphas[:, t] += alphas[:, t - 1].gather(1, betas[:, t])

        # Reconstuction of the most likely sequence.
        predictions = torch.zeros_like(valid_mask, dtype=torch.int64)
        predictions[:, -1] = torch.argmax(alphas[:, -1], dim=-1)
        for t in reversed(range(logits.shape[1] - 1)):
            predictions[:, t] = betas[:, t + 1].gather(1, predictions[:, t + 1].unsqueeze(-1)).squeeze(-1)

        return predictions

    @torch.inference_mode()
    def predict(self, dataset: Dataset, dataloader: torch.utils.data.DataLoader) -> tuple[list[list[tuple[int, int, int]]], list[list[tuple[int, str, int]]]]:
        self.eval()

        results, results_zeros, entities = [], [], 0
        doc_mentions, doc_subwords = [], 0
        for b_subwords, b_word_indices in minnt.ProgressLogger(dataloader, f"Predicting {dataset._path}"):
            b_subwords, b_word_indices = b_subwords.to(self.device), b_word_indices.to(self.device)
            b_size = b_word_indices.shape[0]

            # Compute tag logits
            b_embeddings, b_zero_embeddings, b_tag_logits, b_zdeprel_logits = self(b_subwords, b_word_indices)
            b_tags = self.decode_mentions(b_tag_logits, b_word_indices[:, :-1] >= 0)
            b_zdeprels = torch.argmax(b_zdeprel_logits, dim=-1)
            del b_tag_logits, b_zdeprel_logits

            b_word_indices, b_tags = b_word_indices.numpy(force=True), b_tags.numpy(force=True)
            b_previous, b_mentions, b_refs = [], [], []
            for b in range(b_size):
                word_indices, tags, zdeprels = b_word_indices[b, b_word_indices[b] >= 0], b_tags[b, b_word_indices[b, 1:] >= 0], b_zdeprels[b, b_word_indices[b, 1:] >= 0]
                if word_indices[0] == 2:
                    doc_mentions, doc_subwords = [], 0

                # Decode mentions
                mentions, stack = [], []
                for i, tag in enumerate(self._tags[tag % len(self._tags)] for tag in tags):
                    for command in tag.split(","):
                        if command == "PUSH":
                            stack.append(i)
                        elif command.startswith("POP:"):
                            j = int(command.removeprefix("POP:"))
                            if len(stack):
                                j = len(stack) - (j if j <= len(stack) else 1)
                                mentions.append((stack.pop(j), i, None))
                        elif command:
                            raise ValueError(f"Unknown command '{command}'")
                while len(stack):
                    mentions.append((stack.pop(), len(tags) - 1, None))

                # Decode zero mentions
                for i, zdeprel in enumerate(zdeprels):
                    for j in range(self._args.zeros_per_parent):
                        if zdeprel[j] == Dataset.ZDEPREL_NONE:
                            break
                        mentions.append((i, -j - 1, self._zdeprels[zdeprel[j]]))

                # Prepare inputs for antecedent prediction
                mentions = sorted(set(mentions), key=lambda x: (x[0], -x[1]))
                offset = doc_subwords - (word_indices[0] - 2)
                results.append([]), results_zeros.append([]), b_previous.append([]), b_mentions.append([]), b_refs.append([])
                for doc_mention in doc_mentions:
                    if doc_mention[0] < offset:
                        continue
                    b_previous[-1].append([doc_mention[0] - offset + 1, doc_mention[1] if doc_mention[1] < 0 else doc_mention[1] - offset + 1])
                    b_refs[-1].append(doc_mention[2])
                for mention in mentions:
                    if mention[2] is not None:
                        result_mention = [mention[0], mention[2], None]
                        results_zeros[-1].append(result_mention)
                    else:
                        result_mention = [mention[0], mention[1], None]
                        results[-1].append(result_mention)
                    b_refs[-1].append(result_mention)
                    b_mentions[-1].append([word_indices[mention[0]], mention[1] if mention[1] < 0 else word_indices[mention[1]]])
                    doc_mentions.append([doc_subwords + word_indices[mention[0]] - word_indices[0],
                                         mention[1] if mention[1] < 0 else doc_subwords + word_indices[mention[1]] - word_indices[0], result_mention])
                doc_subwords += word_indices[-1] - word_indices[0]

            # Decode antecedents
            if sum(len(mentions) for mentions in b_mentions) == 0:
                continue
            b_all_mentions = [previous + mentions for previous, mentions in zip(b_previous, b_mentions)]
            b_antecedents = self.compute_antecedents(
                b_embeddings, b_zero_embeddings,
                torch.nn.utils.rnn.pad_sequence([torch.as_tensor(m, dtype=torch.int64).view(-1, 2) for m in b_all_mentions], batch_first=True, padding_value=0).to(self.device),
                torch.nn.utils.rnn.pad_sequence([torch.as_tensor(m, dtype=torch.int64).view(-1, 2) for m in b_mentions], batch_first=True, padding_value=0).to(self.device),
            ).numpy(force=True)
            del b_embeddings

            for b in range(b_size):
                len_prev, mentions, refs, antecedents = len(b_previous[b]), b_mentions[b], b_refs[b], b_antecedents[b]
                for i in range(len(mentions)):
                    j = i - 1
                    while j >= 0 and mentions[j][0] == mentions[i][0]:
                        antecedents[i, j + len_prev] = antecedents[i, i + len_prev] - 1
                        j -= 1
                    j = np.argmax(antecedents[i, :i + len_prev + 1])
                    if j == i + len_prev:
                        entities += 1
                        refs[i + len_prev][2] = entities
                    else:
                        refs[i + len_prev][2] = refs[j][2]

        return results, results_zeros

    def process(self, epoch: int, datasets: list[tuple[Dataset, torch.utils.data.DataLoader]], evaluate: bool) -> None:
        for dataset, dataloader in datasets:
            mentions, zero_mentions = self.predict(dataset, dataloader)
            path = os.path.join(self._args.logdir, f"{os.path.splitext(os.path.basename(dataset._path))[0]}.{epoch:02d}.conllu")
            dataset.save_mentions(path, mentions, zero_mentions)
            if evaluate:
                # You might want to run the evaluation in parallel if you can; we used `sbatch` during development.
                os.system(f"./corefud-score.sh '{dataset._path}' '{path}'")


def main(params: list[str] | None = None) -> None:
    args = parser.parse_args(params)

    # Set the random seed and the number of threads
    minnt.startup(args.seed, args.threads)

    # If supplied, load configuration from a trained model
    if args.load:
        resolved_load_path = args.load[0] if os.path.exists(args.load[0]) else huggingface_hub.snapshot_download(args.load[0])
        with open(os.path.join(resolved_load_path, "options.json"), mode="r") as options_file:
            args = argparse.Namespace(**{k: v for k, v in json.load(options_file).items() if k in [
                "batch_size", "depth", "encoder", "right", "segment", "treebanks"]})
        args = parser.parse_args(params, namespace=args)
        args.load = [resolved_load_path]
        args.logdir = args.exp if args.exp else "."
    else:
        if not args.train:
            raise ValueError("Either --load or --train must be set.")
        args.logdir = os.path.join("logs", "{}{}-{}-{}-{}".format(
            args.exp + (args.exp and "-"),
            os.path.splitext(os.path.basename(globals().get("__file__", "notebook")))[0],
            os.environ.get("SLURM_JOB_ID", ""),
            datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
            ",".join(("{}={}".format(
                re.sub("(.)[^_]*_?", r"\1", k),
                ",".join(re.sub(r"^.*/", "", str(x)) for x in ((v if len(v) <= 1 else [v[0], "..."]) if isinstance(v, list) else [v])),
            ) for k, v in sorted(vars(args).items()) if k not in ["compile", "dev", "test", "exp", "load", "threads"]))
        ))
        print(json.dumps(vars(args), sort_keys=True, ensure_ascii=False, indent=2))

    # Create the tokenizer, using a hack to allow sharing tokenized data among models with the same tokenizers.
    if "t5gemma" in args.encoder:
        tokenizer_name = "google/t5gemma-l-l-ul2"
    elif "umt5" in args.encoder:
        tokenizer_name = "google/umt5-xl"
    elif "mt5" in args.encoder:
        tokenizer_name = "google/mt5-xl"
    else:
        tokenizer_name = args.encoder
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)  # The legacy does not change things, but silences a warning.
    tokenizer.add_special_tokens({"additional_special_tokens": [Dataset.TOKEN_EMPTY] + ([Dataset.TOKEN_CLS] if tokenizer.cls_token_id is None else [])})

    # Load the data
    trains = [Dataset(path, tokenizer) for path in args.treebanks] if args.train else []

    devs = [Dataset(path.replace("-train.conllu", "-minidev.conllu"), tokenizer) for path in ([] if args.dev is None else (args.dev or args.treebanks)) if path]

    tests = [Dataset(path.replace("-train.conllu", "-minitest.conllu"), tokenizer) for path in ([] if args.test is None else (args.test or args.treebanks)) if path]

    if args.load:
        with open(os.path.join(args.load[0], "tags.txt"), mode="r") as tags_file:
            tags = [line.rstrip("\r\n") for line in tags_file]
        with open(os.path.join(args.load[0], "zdeprels.txt"), mode="r") as zdeprels_file:
            zdeprels = [line.rstrip("\r\n") for line in zdeprels_file]
    else:
        tags = Dataset.create_tags(trains)
        zdeprels = Dataset.create_zdeprels(trains)
    tags_map = {tag: i for i, tag in enumerate(tags)}
    zdeprels_map = {zdeprel: i for i, zdeprel in enumerate(zdeprels)}

    # Create dataloaders
    if args.train:
        train = TrainDataset([train.dataset(tags_map, zdeprels_map, True, args) for train in trains])
        train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, collate_fn=Dataset.padded_batch(True), sampler=train.sampler(args))
    devs = [(dev, torch.utils.data.DataLoader(
        dev.dataset(tags_map, zdeprels_map, False, args), batch_size=args.batch_size, collate_fn=Dataset.padded_batch(False))) for dev in devs]
    tests = [(test, torch.utils.data.DataLoader(
        test.dataset(tags_map, zdeprels_map, False, args), batch_size=args.batch_size, collate_fn=Dataset.padded_batch(False))) for test in tests]

    model = Model(tokenizer, tags, zdeprels, args)
    if args.load:
        model.load_weights(os.path.join(args.load[0], "model.pt"))
    if args.compile:
        model.compile(dynamic=True)

    if args.train:
        # Create logdir with the source, options, and tags
        os.makedirs(args.logdir)
        shutil.copy2(__file__, os.path.join(args.logdir, os.path.basename(__file__)))
        with open(os.path.join(args.logdir, "options.json"), "w") as json_file:
            json.dump(vars(args), json_file, sort_keys=True, ensure_ascii=False, indent=2)
        with open(os.path.join(args.logdir, "tags.txt"), "w") as tags_file:
            for tag in tags:
                print(tag, file=tags_file)
        with open(os.path.join(args.logdir, "zdeprels.txt"), "w") as zdeprels_file:
            for zdeprel in zdeprels:
                print(zdeprel, file=zdeprels_file)
        # Configure the model and train
        model.configure(train)
        model.fit(train, epochs=args.epochs, callbacks=[
            lambda model, epoch, logs: model.save_weights(f"{args.logdir}/model{epoch:02d}.pt"),
            lambda model, epoch, logs: model.process(epoch, devs, evaluate=True),
            lambda model, epoch, logs: model.process(epoch, tests, evaluate=False),
        ])

    elif args.dev is not None or args.test is not None:
        os.makedirs(args.logdir, exist_ok=True)
        if args.dev is not None:
            model.process(args.epochs, devs, evaluate=True)
        if args.test is not None:
            model.process(args.epochs, tests, evaluate=False)


if __name__ == "__main__":
    main([] if "__file__" not in globals() else None)
