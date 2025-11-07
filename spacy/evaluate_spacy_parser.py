import sys
import subprocess
from pathlib import Path

import spacy
from spacy.tokens import Doc
from conllu import parse_incr
from stanza.utils.conll import CoNLL  # only for writing convenience; or we can write manually


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UDBASE = PROJECT_ROOT / "data" / "udbase" / "UD_Uzbek-UzUDT"
TEST_CONLLU = UDBASE / "uz_uzudt-ud-test.conllu"

SPACY_MODEL_DIR = PROJECT_ROOT / "spacy" / "models" / "uz_uzudt_parser" / "model-best"
RESULTS_DIR = PROJECT_ROOT / "spacy" / "results"
SYSTEM_CONLLU = RESULTS_DIR / "uz_uzudt-ud-test.spacy.system.conllu"

EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval.py"  # official UD eval.py you added


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading spaCy model from {SPACY_MODEL_DIR}")
    nlp = spacy.load(SPACY_MODEL_DIR)

    print(f"Reading gold test from {TEST_CONLLU}")
    with TEST_CONLLU.open(encoding="utf-8") as f:
        gold_sents = list(parse_incr(f))

    # Convert gold sentences to spaCy Docs and run the pipeline
    docs = []
    for sent in gold_sents:
        words = []
        for token in sent:
            if not isinstance(token["id"], int):
                continue
            words.append(token["form"])

        if not words:
            continue

        # create a Doc with the gold tokens (no re-tokenization)
        doc = Doc(nlp.vocab, words=words)
        doc = nlp(doc)
        docs.append(doc)

    # Now convert spaCy Docs back to CoNLL-U format
    # We'll write a new system .conllu file that mirrors the gold structure
    print(f"Writing system predictions to {SYSTEM_CONLLU}")
    with SYSTEM_CONLLU.open("w", encoding="utf-8") as out_f:
        for sent, doc in zip(gold_sents, docs):
            # write metadata (e.g. sent_id, text) from gold
            for key, value in (sent.metadata or {}).items():
                out_f.write(f"# {key} = {value}\n")

            # map spaCy tokens to CoNLL-U rows
            idx = 1
            for gold_token, spacy_token in zip(
                [t for t in sent if isinstance(t["id"], int)],
                doc
            ):
                head_idx = spacy_token.head.i + 1 if spacy_token.head is not spacy_token else 0
                deprel = spacy_token.dep_.lower() if spacy_token.dep_ != "ROOT" else "root"

                # Use spaCy UPOS if available, otherwise underscore
                upos = spacy_token.pos_ if spacy_token.pos_ else "_"

                # Construct a CoNLL-U line
                fields = [
                    str(idx),                         # ID
                    gold_token["form"],              # FORM
                    "_",                             # LEMMA (not predicted here)
                    upos,                            # UPOS
                    "_",                             # XPOS
                    "_",                             # FEATS
                    str(head_idx),                   # HEAD
                    deprel,                          # DEPREL
                    "_",                             # DEPS
                    "_"                              # MISC
                ]
                out_f.write("\t".join(fields) + "\n")
                idx += 1

            out_f.write("\n")

    print("✅ Done writing system CoNLL-U.")

    # Optionally run official UD eval.py if it exists
    if EVAL_SCRIPT.exists():
        print("\nRunning official UD eval.py on spaCy system output...\n")
        cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "-v",
            str(TEST_CONLLU),
            str(SYSTEM_CONLLU),
        ]
        subprocess.run(cmd)
    else:
        print(
            f"⚠️ Could not find eval.py at {EVAL_SCRIPT}. "
            "You can evaluate manually with:\n"
            f"  python scripts/eval.py -v {TEST_CONLLU} {SYSTEM_CONLLU}"
        )


if __name__ == "__main__":
    main()
