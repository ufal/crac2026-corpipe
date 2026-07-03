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
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=str, help="Experiment name")
parser.add_argument("epochs", default=0, nargs="?", type=int, help="Epochs to show")
parser.add_argument("-c", default=None, type=str, help="Compare to another experiment")
args = parser.parse_args()

treebanks = [
    "ca", "cs_pce", "cs_pdt", "cs_pdts", "cu", "de", "en_fan", "en_gum", "en_lit",
    "es", "fr_anc", "fr_dem", "fr_lit", "grc", "hbo", "hi", "hu_kor", "hu_sze",
    "ko", "la", "lt_lcc", "nl", "no_bok", "no_nyn", "pl", "ru", "tr",
]

# Load the data
def load(exp):
    exp_name, exp_suffix = exp, "eval"
    if exp_name.endswith((".e", ".s")):
        exp_name, exp_suffix = exp_name[:-2], f"eval{exp_name[-1]}"
    if "/" not in exp_name: exp_name = f"logs/{exp_name}*"
    results = {}
    for path in sorted(glob.glob(f"{exp_name}/*[0-9].{exp_suffix}")):
        base, epoch, *_ = os.path.basename(path)[:-len(exp_suffix)-1].split(".")
        epoch = int(epoch)
        for treebank in reversed(treebanks):
            if base.startswith(treebank):
                base = treebank
                break
        else:
            raise ValueError(f"Unknown treebank for evaluation '{base}'")
        results.setdefault(base, {})
        if epoch in results[base]:
            raise ValueError(f"Multiple evaluations for '{base}' epoch '{epoch}'")
        with open(path, "r", encoding="utf-8") as eval_file:
            for line in eval_file:
                line = line.rstrip("\r\n")
                if line.startswith("CoNLL score: "):
                    results[base][epoch] = line[13:]
    return results
results = load(args.exp)

# Print them out
def avg(callback, results):
    best_epoch = max(((sum(float(results[t][e]) for t in treebanks) / len(treebanks), e)
                      for e in results.get(treebanks[0], {}) if all(e in results.get(t, {}) for t in treebanks)), default=(None, 0))[1]
    values = [callback(results[t], best_epoch) if t in results else "" for t in treebanks]
    if all(values):
        values.append("{:.2f}".format(sum(float(value) for value in values) / len(values)))
    return values
if args.c:
    others = load(args.c)
    def show(callback):
        xs, ys = avg(callback, results), avg(callback, others)
        return ["\033[{}m{:+7.2f}\033[0m".format(32 if float(x) >= float(y) else 31,
                                                float(x) - float(y)) if x and y else ""
                for x, y in zip(xs, ys)]
else:
    show = lambda callback: avg(callback, results)
def pprint(*values):
    print(*(f"{value:<7}" for value in values), sep="")
pprint("mode", *treebanks, "avg")
pprint("last", *show(lambda res, _: list(res.values())[-1]))
pprint("best", *show(lambda res, best: res.get(best, "")))
pprint("max", *show(lambda res, _: max(list(res.values()), key=float)))
offset = 0 if any(0 in res for res in results.values()) else 1
for epoch in range(offset, offset + args.epochs):
    pprint(epoch, *show(lambda res, _: res.get(epoch, "")))
