#!/usr/bin/env python3
"""D3: PromptBench adv_prompts/*.md -> one CSV."""
import ast
import csv
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw/promptbench")
OUT_CSV = Path("data/interim/promptbench_prompts.csv")

RE_CLEAN = re.compile(r"^Acc:\s*(-?[\d.]+)%,\s*prompt:\s*(.*)$", re.I)
RE_SEM = re.compile(
    r"^Language:\s*([^,]+),\s*acc:\s*(-?[\d.]+)%,\s*prompt:\s*(.*)$", re.I)
RE_ORIG = re.compile(r"^Original prompt:\s*(.*)$")
RE_ATT = re.compile(r"^Attacked prompt:\s*(.*)$")
RE_ACCS = re.compile(
    r"^Original acc:\s*(-?[\d.]+)%,\s*attacked acc:\s*(-?[\d.]+)%,"
    r"\s*dropped acc:\s*(-?[\d.]+)%")

FIELDS = ["id", "text", "label", "source", "language",
          "pb_model", "pb_shot", "pb_task", "pb_attack", "prompt_kind",
          "pb_sem_lang", "original_prompt",
          "original_acc", "attacked_acc", "dropped_acc"]


def decode_bytes_literal(s):
    """b"..." ধরনের Python bytes লেখাকে সাধারণ text-এ আনে।"""
    s = s.strip()
    if len(s) >= 3 and s[0] == "b" and s[1] in "'\"":
        try:
            return ast.literal_eval(s).decode("utf-8", errors="replace")
        except (ValueError, SyntaxError):
            return s[2:-1]
    return s


def make_row(model, shot, task, attack, kind, text, sem_lang="",
             original="", o_acc="", a_acc="", d_acc=""):
    return {"text": text, "label": 0, "source": "promptbench",
            "language": "en", "pb_model": model, "pb_shot": shot,
            "pb_task": task, "pb_attack": attack, "prompt_kind": kind,
            "pb_sem_lang": sem_lang, "original_prompt": original,
            "original_acc": o_acc, "attacked_acc": a_acc,
            "dropped_acc": d_acc}


def parse_file(md):
    out, bad = [], []
    stem = md.stem
    model, shot = stem.split("_", 1)
    task = None
    section = None
    pending = None

    def flush():
        nonlocal pending
        if pending and pending.get("attacked"):
            accs = pending.get("accs", ("", "", ""))
            out.append(make_row(model, shot, task, section, "attacked",
                                pending["attacked"],
                                original=pending["original"],
                                o_acc=accs[0], a_acc=accs[1], d_acc=accs[2]))
        pending = None

    for raw in md.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            continue
        if line.startswith("# "):
            flush()
            name = line[2:].strip()
            if name != stem:
                task = name
                section = None
            continue
        m = RE_SEM.match(line)
        if m:
            flush()
            text = m.group(3).strip()
            if text:
                out.append(make_row(model, shot, task, "semantic", "attacked",
                                    text, sem_lang=m.group(1).strip().lower(),
                                    o_acc=m.group(2)))
            continue
        m = RE_CLEAN.match(line)
        if m:
            flush()
            text = m.group(2).strip()
            if text:
                is_clean = section is None or "prompts" in section
                out.append(make_row(model, shot, task,
                                    "none" if is_clean else section,
                                    "clean" if is_clean else "attacked",
                                    text, o_acc=m.group(1)))
            continue
        m = RE_ORIG.match(line)
        if m:
            flush()
            pending = {"original": m.group(1).strip()}
            continue
        m = RE_ATT.match(line)
        if m and pending is not None:
            pending["attacked"] = decode_bytes_literal(m.group(1))
            continue
        m = RE_ACCS.match(line)
        if m and pending is not None:
            pending["accs"] = m.groups()
            flush()
            continue
        bad.append((md.name, line))
    flush()
    return out, bad


def main():
    files = sorted(RAW_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found in {RAW_DIR.resolve()}")

    rows, bad = [], []
    for md in files:
        r, b = parse_file(md)
        print(f"{md.name:25s} -> {len(r):5d} rows")
        rows += r
        bad += b

    for i, row in enumerate(rows, 1):
        row["id"] = f"D3_{i:05d}"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== SUMMARY =====")
    print(f"Total rows      : {len(rows)}")
    print(f"Unique texts    : {len({r['text'] for r in rows})}")
    print(f"By prompt_kind  : {dict(Counter(r['prompt_kind'] for r in rows))}")
    print(f"By attack       : {dict(Counter(r['pb_attack'] for r in rows))}")
    print(f"By sem language : {dict(Counter(r['pb_sem_lang'] for r in rows if r['pb_sem_lang']))}")
    print(f"Unparsed lines  : {len(bad)}")
    if bad:
        kinds = Counter(line.split(":")[0][:30] for _, line in bad)
        print("Unparsed by prefix:", dict(kinds.most_common(5)))
        for name, line in bad[:5]:
            print(f"    {name}: {line[:100]}")
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
