import os
from pathlib import Path

import spacy
from spacy.tokens import Doc, DocBin
from spacy.cli.init_config import init_config
from spacy.cli.train import train
from conllu import parse_incr


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UDBASE = PROJECT_ROOT / "data" / "udbase" / "UD_Uzbek-UzUDT"

TRAIN_CONLLU = UDBASE / "uz_uzudt-ud-train.conllu"
DEV_CONLLU = UDBASE / "uz_uzudt-ud-dev.conllu"

SPACY_DIR = PROJECT_ROOT / "spacy"
DATA_DIR = SPACY_DIR / "data"
CONFIG_DIR = SPACY_DIR / "configs"
MODELS_DIR = SPACY_DIR / "models" / "uz_uzudt_parser"

TRAIN_SPACY = DATA_DIR / "uz_uzudt_train.spacy"
DEV_SPACY = DATA_DIR / "uz_uzudt_dev.spacy"
CONFIG_PATH = CONFIG_DIR / "uz_uzudt.cfg"


def conllu_to_docbin(conllu_path: Path, nlp) -> DocBin:
    """
    Convert a UD CoNLL-U file to a spaCy DocBin for training.
    Uses gold tokens, UPOS tags, heads, and deprels.
    """
    db = DocBin(store_user_data=True)

    print(f"Reading {conllu_path}")
    with conllu_path.open(encoding="utf-8") as f:
        for sent in parse_incr(f):
            words = []
            upos_tags = []
            heads = []
            deps = []

            # Skip multi-word tokens (id as "3-4") and empty nodes.
            for token in sent:
                if not isinstance(token["id"], int):
                    continue

                words.append(token["form"])
                upos = token.get("upos") or token.get("upostag") or "X"
                upos_tags.append(upos)

                head = token["head"]  # 0 = root; 1-based index otherwise
                heads.append(head)
                deps.append(token.get("deprel") or "dep")

            if not words:
                continue

            # Create a Doc with gold tokenization
            doc = Doc(nlp.vocab, words=words)

            # Assign UPOS tags
            for token, upos in zip(doc, upos_tags):
                token.pos_ = upos
                token.tag_ = upos  # store UPOS also as fine-grained tag

            # Assign heads and dependencies
            for i, token in enumerate(doc):
                head_idx = heads[i]
                dep = deps[i]
                if head_idx == 0:
                    # root: head is the token itself
                    token.head = token
                    token.dep_ = "root"
                else:
                    token.head = doc[head_idx - 1]
                    token.dep_ = dep

            db.add(doc)

    return db


def main():
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Create a blank multilingual pipeline
    nlp = spacy.blank("xx")  # "xx" = multi-language

    # 1. Convert CoNLL-U train/dev to spaCy DocBin
    print("Converting CoNLL-U to spaCy DocBin...")
    train_db = conllu_to_docbin(TRAIN_CONLLU, nlp)
    dev_db = conllu_to_docbin(DEV_CONLLU, nlp)

    print(f"Writing {TRAIN_SPACY}")
    train_db.to_disk(TRAIN_SPACY)
    print(f"Writing {DEV_SPACY}")
    dev_db.to_disk(DEV_SPACY)

    # 2. Create a config if it does not exist yet
    if not CONFIG_PATH.exists():
        print(f"Creating spaCy config at {CONFIG_PATH}")
        cfg = init_config(
            lang="xx",
            pipeline=["tok2vec", "tagger", "parser"],
            optimize="efficiency",
        )
        # Set train/dev paths via config's paths section
        cfg["paths"]["train"] = str(TRAIN_SPACY)
        cfg["paths"]["dev"] = str(DEV_SPACY)

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(cfg.to_str())
    else:
        print(f"Using existing config at {CONFIG_PATH}")

    # 3. Train the model
    print("Starting spaCy training...")
    # overrides ensures we use the correct train/dev files
    train(
        CONFIG_PATH,
        output_path=MODELS_DIR,
        overrides={
            "paths.train": str(TRAIN_SPACY),
            "paths.dev": str(DEV_SPACY),
        },
    )
    print(f"Training finished. Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
