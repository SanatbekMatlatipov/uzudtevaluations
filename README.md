# 🇺🇿 UzUDT Evaluations

This repository contains **training, evaluation, and analysis code** for the paper:

> **UzUDT: Uzbek Universal Dependencies Treebank** (LREC 2026)
> Sanatbek Matlatipov, Mersaid Aripov — National University of Uzbekistan

We present a new gold-standard UD treebank for Uzbek containing **684 sentences (7,582 tokens)** from literary texts, with full manual morphosyntactic annotation. This repo provides the complete evaluation framework for comparing:

- **[Stanza](https://stanfordnlp.github.io/stanza/)** — graph-based BiLSTM + DeepBiaffine parser
- **[spaCy](https://spacy.io/)** — transition-based arc-eager parser

across two embedding strategies:
- **FastText** (static subword embeddings)
- **TahrirchiBERT** (monolingual contextual embeddings)

and two data configurations:
- **UzUDT only** (684 sentences)
- **Merged** (UzUDT + UD_Uzbek-UT, combined training data)

---

## 🗂 Repository Structure

```
uzudtevaluations/
│
├── stanza-train/                    # Stanza training & evaluation pipeline
│   ├── scripts/
│   │   ├── pos_predict_from_udbase.py
│   │   ├── parse_test_with_depparse.py
│   │   ├── eval_pos.py
│   │   ├── eval_upos_by_tag.py
│   │   ├── eval.py
│   │   └── visualization.py
│   ├── config/
│   │   └── config.sh               # Environment variables (UDBASE, WORDVEC_DIR, etc.)
│   ├── data/
│   │   ├── udbase/
│   │   │   └── UD_Uzbek-UzUDT/     # CoNLL-U treebank files
│   │   ├── wordvec/                 # FastText embeddings (cc.uz.300.vec)
│   │   └── processed/              # Preprocessed data for each pipeline stage
│   ├── saved_models/               # Trained Stanza model checkpoints
│   ├── stanza/                     # Modified Stanza source (BiLSTM + biaffine)
│   ├── logs/
│   └── requirements.txt
│
├── spacy_uzbek/                     # spaCy training & evaluation pipeline
│   ├── train.py
│   ├── evaluate.py
│   ├── convert_conllu.py
│   ├── register.py
│   └── setup.py
│
├── data/
│   ├── udbase/
│   │   ├── UD_Uzbek-UzUDT/         # UzUDT treebank (train/dev/test)
│   │   └── UD_Uzbek-UT/            # Existing Uzbek-UT treebank
│   ├── tokenize/                   # Tokenization data & MWT dictionaries
│   └── depparse/                   # Dependency parsing input files
│
├── saved_models/                    # Experiment results & model summaries
│   ├── pos/                        # Tagger summaries, logs, and predictions
│   ├── depparse/                   # Parser summaries, logs, and predictions
│   ├── spacy/                      # spaCy trained models
│   └── tokenize/                   # Tokenizer models
│
├── results/                         # spaCy evaluation result JSONs
├── uzudt_main.tex                   # Paper source (LREC 2026)
└── README.md
```

### Experiment Naming Convention

Model files follow the pattern `uz_{dataset}_{experiment}_{component}`:
- **Dataset**: `uzudt` (UzUDT only) or `combined` (UzUDT + Uzbek-UT merged)
- **Experiment ID**: e.g., `E1.1` (FastText, UzUDT), `E2.1` (TahrirchiBERT, UzUDT), `E1.2` (FastText, Merged), `E2.2` (TahrirchiBERT, Merged)
- **Component**: `tagger` or `parser`

> 🧩 **Note:**
> The `stanza-train/data/wordvec/` directory must contain the Uzbek **FastText word embeddings** file
> (`cc.uz.300.vec`), downloadable from [fastText](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uz.300.vec.gz).
> These pretrained vectors are required for the FastText-based Stanza experiments.

---

## ⚙️ Overview

The experiments evaluate **morphosyntactic tagging** (UPOS, UFeats) and **dependency parsing** (UAS, LAS) under varying low-resource constraints:

| Dimension | Options |
|-----------|---------|
| **Architecture** | Stanza (graph-based) vs. spaCy (transition-based) |
| **Embeddings** | FastText (static) vs. TahrirchiBERT (contextual) |
| **Training Data** | UzUDT only vs. Merged (UzUDT + Uzbek-UT) |

### Key Findings

1. **TahrirchiBERT > FastText**: Contextual embeddings consistently outperform static embeddings across all metrics.
2. **Cross-treebank augmentation**: Merging UzUDT with Uzbek-UT yields ~10–11 point LAS improvement for the graph-based parser.
3. **Architectural trade-off**: spaCy (transition-based) excels at UPOS tagging (89.18%); Stanza (graph-based) vastly outperforms on structural parsing (LAS 63.81% vs. 47.11%).

---

## 🧠 Setup Instructions

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install stanza spacy conllu pandas numpy scikit-learn tqdm
pip install torch transformers   # PyTorch 2.6.0+cu124, Transformers 4.49.0
```

### 2. Stanza Dependencies

```bash
cd stanza-train
pip install -r requirements.txt
```

### 3. spaCy with TahrirchiBERT

The spaCy pipeline uses TahrirchiBERT as its transformer encoder:

```bash
cd spacy_uzbek
pip install -e .
```

### 4. Hardware

Experiments were run on a single **NVIDIA RTX A6000** (48 GB VRAM) with CUDA 12.4.
Software versions: Stanza v1.4.0, spaCy v3.8.11, PyTorch v2.6.0+cu124, Transformers v4.49.0.

---

## 🚀 Workflows

### A. Stanza: POS Tagging & Dependency Parsing

#### 1. Configure Environment

```bash
cd stanza-train
source config/config.sh
```

#### 2. Train & Predict POS Tags

```bash
cd scripts
python3 pos_predict_from_udbase.py
```

#### 3. Evaluate POS Accuracy

```bash
python3 eval_pos.py \
  --gold ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu \
  --system ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu
```

#### 4. Per-Tag UPOS Breakdown

```bash
python3 eval_upos_by_tag.py \
  --gold ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu \
  --system ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu
```

#### 5. Dependency Parsing (Test Set)

```bash
python3 parse_test_with_depparse.py
```

---

### B. spaCy: Joint Transition-Based Pipeline

The spaCy pipeline is a joint, end-to-end model using TahrirchiBERT as its encoder.

#### 1. Train

```bash
cd spacy_uzbek
python train.py
```

#### 2. Evaluate

```bash
python evaluate.py
```

Results are saved to `results/`.

---

## 📊 Parser Performance

Parsing and tagging performance (%) on the **UzUDT test set**, comparing architectures, embeddings, and data configurations:

| **Parser** | **Embeddings** | **Data** | **UPOS** | **UFeats** | **UAS** | **LAS** |
|------------|----------------|----------|----------|------------|---------|---------|
| Stanza | FastText | UzUDT | 79.19 | 66.61 | 69.57 | 51.24 |
| Stanza | FastText | Merged | 80.26 | 66.98 | 72.27 | 62.40 |
| Stanza | TahrirchiBERT | UzUDT | 82.45 | 65.37 | 72.05 | 54.19 |
| Stanza | TahrirchiBERT | Merged | 85.08 | **71.09** | **72.39** | **63.81** |
| spaCy | TahrirchiBERT | UzUDT | 86.50 | 50.55 | 67.72 | 45.35 |
| spaCy | TahrirchiBERT | Merged | **89.18** | 65.48 | 66.81 | 47.11 |

**Abbreviations**: Merged = UzUDT + Uzbek-UT combined training data. Stanza uses a graph-based (DeepBiaffine) parser; spaCy uses a transition-based (arc-eager) parser.

### Interpretation

- **spaCy** achieves the best UPOS accuracy (89.18%) thanks to its joint end-to-end objective, which reinforces tagging signals through the parser loss. Merging data triggered a +14.93 point jump in its UFeats accuracy.
- **Stanza** vastly outperforms spaCy on labeled structural parsing (LAS 63.81% vs. 47.11%), a 16.70-point advantage demonstrating that graph-based global decoding is better for resolving complex, long-distance dependencies in head-final (SOV) Uzbek syntax.
- **Cross-treebank augmentation** provides the most substantial boost to structural parsing, improving Stanza LAS by ~10–11 points.

---

## 🧮 Metrics

| Metric | Description |
|---------|-------------|
| **UPOS** | Universal Part-of-Speech tagging accuracy |
| **UFeats** | Morphological features accuracy (exact match of full feature bundle) |
| **UAS** | Unlabeled Attachment Score (correct head only) |
| **LAS** | Labeled Attachment Score (correct head + dependency label) |

---

## 📚 References

1. Qi, P. et al. (2020). *Stanza: A Python Natural Language Processing Toolkit for Many Human Languages.* ACL.
2. Honnibal, M. & Montani, I. (2017). *spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.*
3. Bojanowski, P. et al. (2017). *Enriching Word Vectors with Subword Information.* TACL.
4. Mamasaidov et al. (2023). *TahrirchiBERT: A monolingual BERT model for Uzbek.*
5. de Marneffe, M.-C. et al. (2021). *Universal Dependencies.* Computational Linguistics.
6. Nivre, J. et al. (2020). *Universal Dependencies v2: An evergrowing multilingual treebank collection.*

---

## 👤 Authors

**Sanatbek Matlatipov** & **Mersaid Aripov**
National University of Uzbekistan named after Mirzo Ulugbek
📧 {s.matlatipov, mirsaid.aripov}@nuu.uz

---

## 📄 License

The UzUDT treebank is distributed under **CC BY-SA 4.0** (Creative Commons Attribution–ShareAlike 4.0).

> **Citation:**
> ```bibtex
> @inproceedings{Matlatipov2026UzUDT,
>   title     = {UzUDT: Uzbek Universal Dependencies Treebank},
>   author    = {Matlatipov, Sanatbek and Aripov, Mersaid},
>   year      = {2026},
>   booktitle = {Proceedings of the 15th Language Resources and Evaluation Conference (LREC 2026)},
>   publisher = {ELRA}
> }
> ```
