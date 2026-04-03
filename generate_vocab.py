import json
import random
import re
import time
from pathlib import Path

import eng_to_ipa as ipa
import pandas as pd
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from nltk.corpus import wordnet as wn
from wordfreq import top_n_list, zipf_frequency

import nltk


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGET_SIZE = 2000
RANDOM_SEED = 42

WIKI_SOURCES = [
    "https://en.wikipedia.org/wiki/Glossary_of_engineering",
    "https://en.wikipedia.org/wiki/Glossary_of_architecture",
    "https://en.wikipedia.org/wiki/Glossary_of_physics",
    "https://en.wikipedia.org/wiki/Glossary_of_chemistry_terms",
    "https://en.wikipedia.org/wiki/Glossary_of_computer_science",
    "https://en.wikipedia.org/wiki/Glossary_of_probability_and_statistics",
    "https://en.wikipedia.org/wiki/Glossary_of_calculus",
    "https://en.wikipedia.org/wiki/Glossary_of_structural_engineering",
]

ENGINEERING_STEMS = [
    "struct", "stress", "strain", "load", "beam", "column", "frame", "truss",
    "concrete", "steel", "alloy", "material", "polymer", "ceramic", "composite",
    "thermal", "fluid", "flow", "pressure", "turbine", "pump", "valve",
    "circuit", "voltage", "current", "resistance", "signal", "sensor", "control",
    "system", "model", "design", "analysis", "simulation", "optimi", "algorithm",
    "network", "data", "compute", "digital", "analog", "mechan", "dynamic", "static",
    "energy", "power", "efficien", "process", "manufact", "fabric", "weld", "machin",
    "geotech", "seismic", "foundation", "vibration", "acoustic", "aero", "robot",
    "battery", "semicon", "transistor", "micro", "nano", "quantum", "biome",
]

TECHNICAL_SUFFIXES = (
    "ology", "graphy", "metry", "lysis", "genic", "scope", "meter", "tron", "phage",
    "phase", "static", "dynamics", "thermal", "resistance", "conductive", "elastic",
)

STOPWORDS = {
    "about", "above", "after", "again", "against", "among", "around", "because", "before",
    "below", "between", "could", "every", "first", "found", "great", "group", "having",
    "house", "human", "large", "learn", "other", "place", "point", "small", "sound",
    "state", "still", "study", "their", "there", "these", "thing", "those", "three",
    "under", "water", "where", "which", "while", "world", "would", "young",
    "city", "public", "business", "general", "national", "five", "real", "live", "states",
    "government", "community", "question", "information", "university", "assumption",
    "capability", "consequence", "graduation", "obligation", "submission", "accommodation",
    "ambulance", "testament", "theaters", "heather", "nomination", "optimistic",
}

MEANING_HINTS = {
    "stress": "応力",
    "strain": "ひずみ",
    "load": "荷重",
    "beam": "梁",
    "column": "柱",
    "concrete": "コンクリート",
    "steel": "鋼",
    "alloy": "合金",
    "thermal": "熱",
    "heat": "熱",
    "fluid": "流体",
    "flow": "流れ",
    "pressure": "圧力",
    "circuit": "回路",
    "voltage": "電圧",
    "current": "電流",
    "signal": "信号",
    "sensor": "センサー",
    "control": "制御",
    "design": "設計",
    "analysis": "解析",
    "simulation": "シミュレーション",
    "algorithm": "アルゴリズム",
    "network": "ネットワーク",
    "energy": "エネルギー",
    "power": "電力・動力",
    "efficiency": "効率",
    "material": "材料",
    "mechan": "機械",
    "dynam": "動力学",
    "stat": "静力学・統計",
    "seismic": "地震",
    "foundation": "基礎",
    "robot": "ロボット",
    "battery": "電池",
    "semicon": "半導体",
    "nano": "ナノ",
}

EXAMPLE_TEMPLATES = [
    "The {word} is critical for accurate engineering analysis.",
    "Engineers validated the {word} before finalizing the design.",
    "The report quantifies {word} under realistic loading conditions.",
    "This study improves {word} in a large-scale system.",
    "The team monitored {word} to prevent structural failure.",
    "Optimizing {word} reduced material use and energy demand.",
]


def safe_word(term: str) -> str:
    w = term.strip().lower()
    w = re.sub(r"\[[^\]]*\]", "", w)
    w = re.sub(r"\([^)]*\)", "", w)
    w = w.replace("–", "-").replace("—", "-")
    w = re.sub(r"[^a-z\- ]", "", w)
    w = re.sub(r"\s+", " ", w).strip()
    return w


def is_usable(word: str) -> bool:
    if not word:
        return False
    if len(word.split()) > 3:
        return False
    if len(word) < 3 or len(word) > 24:
        return False
    if word.startswith("list") or word.endswith("list"):
        return False
    if word.count("-") > 1:
        return False
    if not re.match(r"^[a-z][a-z\- ]*$", word):
        return False
    return True


def is_technical_like(word: str) -> bool:
    if any(stem in word for stem in ENGINEERING_STEMS):
        return True
    if any(word.endswith(sfx) for sfx in TECHNICAL_SUFFIXES):
        return True
    if " " in word and any(stem in word for stem in ENGINEERING_STEMS[:40]):
        return True
    return False


def fetch_wiki_terms() -> set[str]:
    terms = set()
    session = requests.Session()
    for url in WIKI_SOURCES:
        try:
            res = session.get(url, timeout=20)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            for li in soup.select("div.mw-parser-output li"):
                txt = safe_word(li.get_text(" ", strip=True))
                if is_usable(txt):
                    terms.add(txt)
        except Exception:
            continue
    return terms


def collect_freq_terms() -> set[str]:
    terms = set()
    common = top_n_list("en", 50000)
    for w in common:
        lw = safe_word(w)
        if not is_usable(lw):
            continue
        if is_technical_like(lw):
            terms.add(lw)
    return terms


def collect_wordnet_technical_terms() -> set[str]:
    terms = set()
    for lemma in wn.all_lemma_names(pos="n"):
        lw = safe_word(lemma)
        if not is_usable(lw) or lw in STOPWORDS:
            continue
        if is_technical_like(lw):
            terms.add(lw)

    for lemma in wn.all_lemma_names(pos="a"):
        lw = safe_word(lemma)
        if not is_usable(lw) or lw in STOPWORDS:
            continue
        if is_technical_like(lw):
            terms.add(lw)
    return terms


def infer_pos(word: str) -> str:
    synsets = wn.synsets(word)
    if synsets:
        pos = synsets[0].pos()
        return {
            "n": "noun",
            "v": "verb",
            "a": "adjective",
            "s": "adjective",
            "r": "adverb",
        }.get(pos, "noun")
    if word.endswith(("tion", "ment", "ness", "ity", "ance", "ence")):
        return "noun"
    if word.endswith(("ize", "ise", "ate", "ify")):
        return "verb"
    if word.endswith(("al", "ive", "ous", "ic", "ary", "able", "ible", "ant", "ent")):
        return "adjective"
    return "noun"


def infer_meaning_ja(word: str) -> str:
    for k, v in MEANING_HINTS.items():
        if k in word:
            return f"{v}に関する工学用語"
    # Fallback: translate the single word only (cheap and stable enough)
    try:
        translated = GoogleTranslator(source="en", target="ja").translate(word)
        if translated:
            return translated
    except Exception:
        pass
    return "工学・科学分野の重要語"


def infer_ipa(word: str) -> str:
    p = ipa.convert(word)
    if not p or "*" in p:
        return f"/{word}/"
    return f"/{p}/"


def score_word(word: str) -> float:
    bonus = 0.0
    for stem in ENGINEERING_STEMS:
        if stem in word:
            bonus += 0.2
    return zipf_frequency(word, "en") + bonus


def build_dataset(words: list[str]) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    rows = []
    for i, w in enumerate(words, start=1):
        pos = infer_pos(w)
        meaning = infer_meaning_ja(w)
        ex = random.choice(EXAMPLE_TEMPLATES).format(word=w)
        rows.append(
            {
                "id": i,
                "word": w,
                "ipa": infer_ipa(w),
                "pos": pos,
                "meaning_ja": meaning,
                "example_en": ex,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    nltk.download("wordnet", quiet=True)

    candidates = set()
    candidates.update(fetch_wiki_terms())
    candidates.update(collect_freq_terms())
    candidates.update(collect_wordnet_technical_terms())

    # Add a compact high-value engineering seed set to ensure coverage.
    seed = {
        "algorithm", "analysis", "architecture", "assembly", "attenuation", "bandwidth", "bearing",
        "buckling", "cad", "cam", "calibration", "cantilever", "capacitance", "catalyst", "cavity",
        "compliance", "compressive", "conductivity", "continuity", "corrosion", "curvature", "damping",
        "deflection", "deformation", "dielectric", "diffusion", "dimension", "ductility", "efficiency",
        "elasticity", "electrode", "elongation", "entropy", "equilibrium", "fatigue", "filtration",
        "friction", "frequency", "hardness", "hydraulics", "impedance", "inductance", "inertia",
        "laminar", "lattice", "manifold", "modulus", "moment", "optimization", "oscillation",
        "permeability", "plasticity", "porosity", "precision", "prototype", "resilience", "rigidity",
        "scalability", "seismic", "shear", "stiffness", "substrate", "tolerance", "torsion", "viscosity",
        "welding", "workload", "yield", "zonation",
    }
    candidates.update(seed)

    filtered = [w for w in candidates if is_usable(w) and is_technical_like(w)]
    filtered = sorted(set(filtered), key=lambda x: (-score_word(x), x))

    if len(filtered) < TARGET_SIZE:
        extra = [safe_word(w) for w in top_n_list("en", 200000)]
        for lw in extra:
            if not is_usable(lw) or lw in STOPWORDS:
                continue
            if lw in filtered:
                continue
            if is_technical_like(lw):
                filtered.append(lw)
            if len(filtered) >= TARGET_SIZE:
                break

    selected = filtered[:TARGET_SIZE]

    df = build_dataset(selected)
    csv_path = DATA_DIR / "engineering_vocab_2000.csv"
    json_path = DATA_DIR / "engineering_vocab_2000.json"
    md_path = DATA_DIR / "engineering_vocab_2000_table.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| English Word | IPA | Part of Speech | 日本語の意味 | Example Sentence |\n")
        f.write("|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(
                f"| {r['word']} | {r['ipa']} | {r['pos']} | {r['meaning_ja']} | {r['example_en']} |\\n"
            )

    print(f"Generated: {len(df)} words")
    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
