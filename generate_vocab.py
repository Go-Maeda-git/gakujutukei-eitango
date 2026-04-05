import random
import re
from pathlib import Path

import eng_to_ipa as ipa
import pandas as pd
import requests
import alkana
from bs4 import BeautifulSoup
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
    "apology", "microsoft", "flowers", "flower", "story", "history", "president",
}

ADVANCED_EXAM_WORDS = {
    # Verbs
    "accelerate": "verb", "accommodate": "verb", "accumulate": "verb", "activate": "verb",
    "adapt": "verb", "adhere": "verb", "aggregate": "verb", "allocate": "verb",
    "amplify": "verb", "anticipate": "verb", "approximate": "verb", "assemble": "verb",
    "attenuate": "verb", "automate": "verb", "calibrate": "verb", "cohere": "verb",
    "collide": "verb", "compensate": "verb", "compile": "verb", "compress": "verb",
    "concentrate": "verb", "condense": "verb", "conduct": "verb", "configure": "verb",
    "consolidate": "verb", "constrain": "verb", "construct": "verb", "consume": "verb",
    "contaminate": "verb", "converge": "verb", "convert": "verb", "correlate": "verb",
    "crystallize": "verb", "curtail": "verb", "decelerate": "verb", "decompose": "verb",
    "deform": "verb", "degrade": "verb", "delineate": "verb", "demonstrate": "verb",
    "denote": "verb", "derive": "verb", "designate": "verb", "detect": "verb",
    "deviate": "verb", "diffuse": "verb", "diminish": "verb", "disperse": "verb",
    "dissipate": "verb", "distort": "verb", "diverge": "verb", "eliminate": "verb",
    "embed": "verb", "emit": "verb", "enclose": "verb", "enhance": "verb",
    "enumerate": "verb", "equate": "verb", "evaporate": "verb", "evolve": "verb",
    "exceed": "verb", "exclude": "verb", "expand": "verb", "exploit": "verb",
    "expose": "verb", "facilitate": "verb", "fluctuate": "verb", "formulate": "verb",
    "fuse": "verb", "generate": "verb", "govern": "verb", "homogenize": "verb",
    "hypothesize": "verb", "illuminate": "verb", "immerse": "verb", "implement": "verb",
    "induce": "verb", "infer": "verb", "inhibit": "verb", "inject": "verb",
    "insert": "verb", "integrate": "verb", "interfere": "verb", "interpolate": "verb",
    "invert": "verb", "isolate": "verb", "justify": "verb", "maintain": "verb",
    "manipulate": "verb", "maximize": "verb", "mediate": "verb", "migrate": "verb",
    "minimize": "verb", "mitigate": "verb", "modify": "verb", "monitor": "verb",
    "neutralize": "verb", "normalize": "verb", "notify": "verb", "optimize": "verb",
    "oscillate": "verb", "partition": "verb", "penetrate": "verb", "perceive": "verb",
    "persist": "verb", "perturb": "verb", "postulate": "verb", "predict": "verb",
    "preserve": "verb", "process": "verb", "propagate": "verb", "quantify": "verb",
    "radiate": "verb", "randomize": "verb", "react": "verb", "reconcile": "verb",
    "reconstruct": "verb", "redistribute": "verb", "refine": "verb", "regulate": "verb",
    "reinforce": "verb", "replicate": "verb", "resolve": "verb", "restrain": "verb",
    "retain": "verb", "retrieve": "verb", "reverse": "verb", "rotate": "verb",
    "saturate": "verb", "simulate": "verb", "specify": "verb", "stabilize": "verb",
    "stimulate": "verb", "substitute": "verb", "suppress": "verb", "sustain": "verb",
    "synthesize": "verb", "transform": "verb", "transmit": "verb", "truncate": "verb",
    "validate": "verb", "verify": "verb", "withstand": "verb",
    # Adjectives
    "abrupt": "adjective", "accurate": "adjective", "adjacent": "adjective", "adverse": "adjective",
    "analog": "adjective", "anisotropic": "adjective", "applicable": "adjective",
    "arbitrary": "adjective", "asymptotic": "adjective", "autonomous": "adjective",
    "axial": "adjective", "balanced": "adjective", "brittle": "adjective", "causal": "adjective",
    "cohesive": "adjective", "coincident": "adjective", "coherent": "adjective",
    "compatible": "adjective", "compressive": "adjective", "computational": "adjective",
    "concurrent": "adjective", "conductive": "adjective", "confined": "adjective",
    "consistent": "adjective", "constitutive": "adjective", "continuous": "adjective",
    "controllable": "adjective", "conventional": "adjective", "corresponding": "adjective",
    "critical": "adjective", "cumulative": "adjective", "cyclic": "adjective",
    "deformable": "adjective", "dense": "adjective", "deterministic": "adjective",
    "differential": "adjective", "dimensionless": "adjective", "discrete": "adjective",
    "distributed": "adjective", "dominant": "adjective", "ductile": "adjective",
    "dynamic": "adjective", "elastic": "adjective", "electromagnetic": "adjective",
    "empirical": "adjective", "equivalent": "adjective", "exponential": "adjective",
    "feasible": "adjective", "finite": "adjective", "frictional": "adjective",
    "functional": "adjective", "homogeneous": "adjective", "hybrid": "adjective",
    "identical": "adjective", "ideal": "adjective", "incremental": "adjective",
    "independent": "adjective", "induced": "adjective", "inert": "adjective",
    "inherent": "adjective", "initial": "adjective", "integral": "adjective",
    "interactive": "adjective", "intrinsic": "adjective", "isotropic": "adjective",
    "iterative": "adjective", "laminar": "adjective", "lateral": "adjective",
    "linear": "adjective", "logarithmic": "adjective", "longitudinal": "adjective",
    "macroscopic": "adjective", "marginal": "adjective", "mechanical": "adjective",
    "microscopic": "adjective", "minimal": "adjective", "modular": "adjective",
    "multivariate": "adjective", "mutual": "adjective", "nonlinear": "adjective",
    "numerical": "adjective", "objective": "adjective", "optimal": "adjective",
    "orthogonal": "adjective", "parallel": "adjective", "parametric": "adjective",
    "periodic": "adjective", "peripheral": "adjective", "plastic": "adjective",
    "porous": "adjective", "predictive": "adjective", "preliminary": "adjective",
    "primary": "adjective", "principal": "adjective", "probabilistic": "adjective",
    "progressive": "adjective", "proportional": "adjective", "prospective": "adjective",
    "radial": "adjective", "random": "adjective", "reciprocal": "adjective",
    "redundant": "adjective", "relative": "adjective", "reliable": "adjective",
    "residual": "adjective", "resilient": "adjective", "robust": "adjective",
    "rotational": "adjective", "scalable": "adjective", "selective": "adjective",
    "sequential": "adjective", "shear": "adjective", "singular": "adjective",
    "spatial": "adjective", "stable": "adjective", "stochastic": "adjective",
    "structural": "adjective", "subsequent": "adjective", "symmetric": "adjective",
    "synchronous": "adjective", "tangential": "adjective", "temporal": "adjective",
    "tensile": "adjective", "thermal": "adjective", "transient": "adjective",
    "uniform": "adjective", "unilateral": "adjective", "variable": "adjective",
    "vectorial": "adjective", "viscous": "adjective", "volatile": "adjective",
}

NOUN_EXAMPLE_TEMPLATES = [
    "The {word} plays a central role in this engineering model.",
    "The experiment quantified {word} under controlled boundary conditions.",
    "Design decisions were validated using {word} from the simulation.",
]

VERB_EXAMPLE_TEMPLATES = [
    "Engineers {word} the parameters to satisfy safety constraints.",
    "The control algorithm can {word} system instability in real time.",
    "Researchers {word} the model to match experimental observations.",
]

ADJECTIVE_EXAMPLE_TEMPLATES = [
    "The structure remained {word} during cyclic loading tests.",
    "A {word} approximation reduced computational cost significantly.",
    "The design requires a {word} response across operating conditions.",
]

ADVERB_EXAMPLE_TEMPLATES = [
    "The solver converged {word} after mesh refinement.",
    "The signal was {word} amplified in the measurement chain.",
    "Loads were {word} distributed across the frame members.",
]

TRANSLATION_CACHE: dict[str, str] = {}

BASE_JA_MEANINGS = {
    "accelerate": "加速する",
    "adapt": "適応する",
    "aggregate": "集約する",
    "amplify": "増幅する",
    "attenuate": "減衰させる",
    "calibrate": "校正する",
    "compress": "圧縮する",
    "conduct": "伝導する",
    "configure": "構成する",
    "constrain": "制約する",
    "converge": "収束する",
    "deform": "変形させる",
    "derive": "導出する",
    "detect": "検出する",
    "diffuse": "拡散する",
    "dissipate": "散逸する",
    "diverge": "発散する",
    "enhance": "高める",
    "equate": "等置する",
    "generate": "生成する",
    "implement": "実装する",
    "induce": "誘起する",
    "inhibit": "抑制する",
    "integrate": "統合する",
    "interpolate": "補間する",
    "isolate": "分離する",
    "mitigate": "緩和する",
    "modify": "修正する",
    "monitor": "監視する",
    "optimize": "最適化する",
    "oscillate": "振動する",
    "partition": "分割する",
    "predict": "予測する",
    "preserve": "保持する",
    "propagate": "伝搬する",
    "quantify": "定量化する",
    "regulate": "制御する",
    "reinforce": "補強する",
    "replicate": "再現する",
    "simulate": "シミュレーションする",
    "stabilize": "安定化する",
    "suppress": "抑制する",
    "sustain": "維持する",
    "synthesize": "合成する",
    "transform": "変換する",
    "transmit": "伝達する",
    "validate": "妥当性を確認する",
    "verify": "検証する",
    "withstand": "耐える",
    "algorithm": "アルゴリズム",
    "analysis": "解析",
    "circuit": "回路",
    "conductivity": "導電率",
    "deformation": "変形",
    "diffusion": "拡散",
    "ductility": "延性",
    "elasticity": "弾性",
    "fatigue": "疲労",
    "friction": "摩擦",
    "frequency": "周波数",
    "impedance": "インピーダンス",
    "inertia": "慣性",
    "laminar": "層流の",
    "modulus": "弾性率",
    "optimization": "最適化",
    "oscillation": "振動",
    "permeability": "透過率",
    "plasticity": "塑性",
    "porosity": "空隙率",
    "resilience": "復元性",
    "seismic": "地震の",
    "shear": "せん断",
    "stiffness": "剛性",
    "torsion": "ねじり",
    "viscosity": "粘性",
    "robust": "頑健な",
    "stable": "安定した",
    "dynamic": "動的な",
    "static": "静的な",
    "nonlinear": "非線形の",
    "probabilistic": "確率論的な",
    "structural": "構造の",
    "tensile": "引張の",
    "thermal": "熱の",
    "transient": "過渡的な",
    "continuous": "連続的な",
    "autonomous": "自律的な",
    "proportional": "比例する",
    "electromagnetic": "電磁の",
    "porous": "多孔質の",
    "unilateral": "片側の",
    "homogeneous": "均質な",
    "cyclic": "周期的な",
    "stochastic": "確率的な",
    "synchronous": "同期した",
    "orthogonal": "直交する",
    "viscous": "粘性のある",
    "multivariate": "多変量の",
    "parametric": "パラメトリックな",
    "asymptotic": "漸近的な",
    "dimensionless": "無次元の",
    "deformable": "変形可能な",
    "microwave": "マイクロ波",
    "ecosystem": "生態系",
    "psychology": "心理学",
    "photography": "写真術",
    "efficiently": "効率的に",
    "sociology": "社会学",
    "semiconductor": "半導体",
    "nano": "ナノ",
    "ceramics": "セラミックス",
    "physiology": "生理学",
    "overflow": "オーバーフロー",
    "pathology": "病理学",
    "metadata": "メタデータ",
    "microscopy": "顕微鏡法",
    "microbiology": "微生物学",
    "overflowing": "あふれている",
    "biotechnology": "バイオテクノロジー",
}

STEM_MEANINGS = {
    "struct": "構造",
    "stress": "応力",
    "strain": "ひずみ",
    "load": "荷重",
    "beam": "梁",
    "column": "柱",
    "concrete": "コンクリート",
    "steel": "鋼",
    "alloy": "合金",
    "material": "材料",
    "thermal": "熱",
    "fluid": "流体",
    "pressure": "圧力",
    "circuit": "回路",
    "voltage": "電圧",
    "current": "電流",
    "resistance": "抵抗",
    "sensor": "センサ",
    "control": "制御",
    "design": "設計",
    "analysis": "解析",
    "simulation": "シミュレーション",
    "algorithm": "アルゴリズム",
    "network": "ネットワーク",
    "digital": "デジタル",
    "analog": "アナログ",
    "mechan": "機械",
    "dynamic": "動的",
    "static": "静的",
    "energy": "エネルギー",
    "power": "電力",
    "process": "プロセス",
    "manufact": "製造",
    "vibration": "振動",
    "seismic": "地震",
    "foundation": "基礎",
    "robot": "ロボット",
    "battery": "電池",
    "quantum": "量子",
    "biome": "生体力学",
}

TOKEN_JA_MEANINGS = {
    "system": "システム",
    "data": "データ",
    "model": "モデル",
    "computer": "コンピュータ",
    "machine": "機械",
    "technology": "技術",
    "signal": "信号",
    "frame": "フレーム",
    "phase": "位相",
    "efficiency": "効率",
    "database": "データベース",
    "framework": "枠組み",
    "pump": "ポンプ",
    "valve": "バルブ",
    "network": "ネットワーク",
    "pressure": "圧力",
    "energy": "エネルギー",
    "process": "過程",
    "material": "材料",
    "structure": "構造",
    "foundation": "基礎",
    "stress": "応力",
    "strain": "ひずみ",
    "flow": "流れ",
    "power": "電力",
    "analysis": "解析",
    "design": "設計",
    "control": "制御",
    "current": "電流",
    "voltage": "電圧",
    "resistance": "抵抗",
    "simulation": "シミュレーション",
    "algorithm": "アルゴリズム",
    "dynamic": "動的",
    "static": "静的",
    "sensor": "センサ",
    "circuit": "回路",
    "ductile": "延性のある",
    "elastic": "弾性のある",
    "thermal": "熱の",
    "seismic": "地震の",
}

BAD_NON_TECH_SUBSTRINGS = {
    "flower",
    "butter",
    "story",
    "ethic",
}

TOKYO_U_ENGINEERING_WORDS = {
    "abate", "aberration", "acumen", "adept", "adjacent", "aggregate", "allocate", "ambient",
    "analogy", "anomaly", "antagonistic", "arbitrary", "ascertain", "coherent", "coincide",
    "commensurate", "complement", "composite", "confer", "constrain", "contiguous", "contingent",
    "conventional", "correlate", "criterion", "cumulative", "delineate", "detrimental", "discrete",
    "disseminate", "divergent", "elucidate", "empirical", "enumerate", "equilibrium", "explicit",
    "feasible", "fluctuate", "formidable", "frictional", "gradient", "heuristic", "homogeneous",
    "hypothesis", "implicit", "incentive", "inherent", "integral", "intermittent", "intrinsic",
    "invoke", "linear", "logarithmic", "manifold", "marginal", "mitigate", "modular", "momentum",
    "notion", "nuance", "optimum", "orthogonal", "paradigm", "parameter", "peripheral",
    "pertinent", "plausible", "precedent", "preliminary", "presumptive", "probabilistic",
    "propagate", "proportional", "protocol", "quantitative", "radial", "rational", "reciprocal",
    "redundant", "refine", "regime", "robust", "salient", "sequential", "sophisticated",
    "spatial", "spectrum", "stochastic", "subtle", "sufficient", "symmetric", "temporal",
    "tentative", "threshold", "topology", "transient", "truncate", "uniform", "validate",
    "variant", "viable", "vulnerable",
}

EXTRA_ENGINEERING_TERMS = {
    "thermodynamics", "thermochemistry", "thermoelasticity", "electrochemistry", "electrodynamics",
    "electromagnetics", "electromechanics", "electrostatics", "microstructure", "macrostructure",
    "metallurgy", "tribology", "rheology", "continuum", "continuity", "compressibility",
    "permittivity", "permeance", "capacitance", "inductance", "susceptibility", "admittance",
    "reactance", "resistivity", "conductance", "diffusivity", "transmissivity", "refractivity",
    "reflectivity", "absorptivity", "emissivity", "plasticization", "crystallization", "solidification",
    "vaporization", "condensation", "nucleation", "granularity", "anisotropy", "isotropy",
    "orthotropy", "viscoelasticity", "elastoplasticity", "incompressibility", "compressive", "tensility",
    "torsional", "flexural", "buckling", "fractography", "fracture", "microfabrication", "nanofabrication",
    "photolithography", "etching", "deposition", "sintering", "annealing", "quenching", "tempering",
    "machinability", "weldability", "formability", "printability", "scattering", "interference",
    "diffraction", "polarization", "modulation", "demodulation", "multiplexing", "sampling",
    "quantization", "linearization", "stabilization", "regularization", "discretization", "normalization",
    "optimization", "parameterization", "identifiability", "observability", "controllability",
    "synchronization", "desynchronization", "parallelization", "vectorization", "serialization",
    "interoperability", "scalability", "reliability", "availability", "maintainability", "traceability",
    "repeatability", "reproducibility", "calibration", "validation", "verification", "benchmarking",
    "instrumentation", "actuation", "localization", "navigation", "trajectory", "kinematics", "dynamics",
    "statics", "hydrodynamics", "aerodynamics", "thermofluid", "biomechanics", "geomechanics",
    "seismology", "geostatics", "hydraulics", "pneumatics", "servo", "feedforward", "feedback",
    "state-space", "eigenvalue", "eigenvector", "covariance", "correlation", "regression",
    "approximation", "interpolation", "extrapolation", "perturbation", "convergence", "divergence",
    "stability", "instability", "transience", "resonance", "damping", "stiffness", "hardening",
    "softening", "fatigue", "creep", "wear", "corrosion", "oxidation", "reduction", "catalysis",
    "adsorption", "desorption", "filtration", "sedimentation", "centrifugation", "distillation",
    "extraction", "purification", "polymerization", "copolymer", "semiconductor", "transconductance",
    "bandgap", "dielectric", "transistor", "photodiode", "optoelectronics", "microcontroller",
    "firmware", "middleware", "throughput", "latency", "bandwidth", "protocol", "checksum",
    "redundancy", "robustness", "fault-tolerance", "cybersecurity", "cryptography", "decryption",
    "encryption", "computation", "algorithmic", "heuristics", "determinism", "probability",
}


def safe_word(term: str) -> str:
    w = term.strip().lower()
    w = re.sub(r"\[[^\]]*\]", "", w)
    w = re.sub(r"\([^)]*\)", "", w)
    w = w.replace("–", "-").replace("—", "-")
    w = re.sub(r"[^a-z\- ]", "", w)
    w = re.sub(r"\s+", " ", w).strip()
    return w


def singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    lemma = wn.morphy(token, "n")
    if lemma:
        return lemma
    if token.endswith("es") and len(token) > 4:
        return token[:-1]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def canonical_word(word: str) -> str:
    parts = [singularize_token(p) for p in re.split(r"[- ]+", word) if p and len(p) > 1]
    return " ".join(parts)


def choose_better_word(a: str, b: str) -> str:
    score_a = score_word(a) + (0.3 if a in ADVANCED_EXAM_WORDS else 0.0) - (0.1 if "-" in a else 0.0)
    score_b = score_word(b) + (0.3 if b in ADVANCED_EXAM_WORDS else 0.0) - (0.1 if "-" in b else 0.0)
    if score_b > score_a:
        return b
    return a


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
    if word in ADVANCED_EXAM_WORDS:
        return True
    if word in STOPWORDS:
        return False
    if zipf_frequency(word, "en") > 6.1 and word not in ADVANCED_EXAM_WORDS:
        return False
    if any(bad in word for bad in BAD_NON_TECH_SUBSTRINGS):
        return False
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
    if word in ADVANCED_EXAM_WORDS:
        return ADVANCED_EXAM_WORDS[word]

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
    if word in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[word]

    if word in BASE_JA_MEANINGS:
        TRANSLATION_CACHE[word] = BASE_JA_MEANINGS[word]
        return BASE_JA_MEANINGS[word]

    hits = []
    for stem, ja in STEM_MEANINGS.items():
        if stem in word and ja not in hits:
            hits.append(ja)
    if hits:
        val = "・".join(hits[:2])
        TRANSLATION_CACHE[word] = val
        return val

    parts = re.split(r"[- ]+", word)
    mapped_parts = []
    for p in parts:
        base = singularize_token(p)

        if base in TOKEN_JA_MEANINGS:
            mapped_parts.append(TOKEN_JA_MEANINGS[base])
        else:
            kana = alkana.get_kana(base)
            if kana:
                mapped_parts.append(kana)

    if mapped_parts:
        val = "・".join(mapped_parts)
        TRANSLATION_CACHE[word] = val
        return val

    synsets = wn.synsets(word)
    if synsets and synsets[0].pos() == "v":
        fallback = "技術的に実行する"
    elif synsets and synsets[0].pos() in {"a", "s"}:
        fallback = "技術的な性質"
    elif synsets and synsets[0].pos() == "r":
        fallback = "技術的な方法"
    else:
        fallback = "技術専門語"

    TRANSLATION_CACHE[word] = fallback
    return fallback


def is_specific_meaning(meaning: str) -> bool:
    return meaning not in {"工学用語", "技術専門語", "工学で用いる動作", "工学で用いる性質", "工学で用いる様態", "技術的に実行する", "技術的な性質", "技術的な方法"}


def has_specific_meaning(word: str) -> bool:
    return is_specific_meaning(infer_meaning_ja(word))


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
    if word in ADVANCED_EXAM_WORDS:
        bonus += 1.8
    return zipf_frequency(word, "en") + bonus


def build_example(word: str, pos: str) -> str:
    if pos == "verb":
        return random.choice(VERB_EXAMPLE_TEMPLATES).format(word=word)
    if pos == "adjective":
        return random.choice(ADJECTIVE_EXAMPLE_TEMPLATES).format(word=word)
    if pos == "adverb":
        return random.choice(ADVERB_EXAMPLE_TEMPLATES).format(word=word)
    return random.choice(NOUN_EXAMPLE_TEMPLATES).format(word=word)


def build_dataset(words: list[str]) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    rows = []
    for i, w in enumerate(words, start=1):
        pos = infer_pos(w)
        meaning = infer_meaning_ja(w)
        ex = build_example(w, pos)
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


def quality_check(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    jp_pat = re.compile(r"[ぁ-んァ-ヶ一-龯]")
    vague_terms = {"工学用語", "技術専門語"}

    for _, r in df.iterrows():
        word = str(r["word"])
        meaning = str(r["meaning_ja"])
        example = str(r["example_en"])

        meaning_ok = meaning not in vague_terms and bool(jp_pat.search(meaning))
        example_contains_word = word.lower() in example.lower()
        example_length_ok = 25 <= len(example) <= 150
        example_style_ok = example.endswith(".")

        checks.append(
            {
                "id": r["id"],
                "word": word,
                "meaning_ok": meaning_ok,
                "example_contains_word": example_contains_word,
                "example_length_ok": example_length_ok,
                "example_style_ok": example_style_ok,
                "overall_ok": bool(meaning_ok and example_contains_word and example_length_ok and example_style_ok),
            }
        )

    return pd.DataFrame(checks)


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
    seed.update(ADVANCED_EXAM_WORDS.keys())
    seed.update(TOKYO_U_ENGINEERING_WORDS)
    seed.update(EXTRA_ENGINEERING_TERMS)
    candidates.update(seed)

    filtered = [w for w in candidates if is_usable(w) and is_technical_like(w) and w not in STOPWORDS]
    filtered = sorted(set(filtered), key=lambda x: (-score_word(x), x))

    deduped_map: dict[str, str] = {}
    for w in filtered:
        key = canonical_word(w)
        if key not in deduped_map:
            deduped_map[key] = w
        else:
            deduped_map[key] = choose_better_word(deduped_map[key], w)
    filtered = list(deduped_map.values())
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

    pos_buckets = {"noun": [], "verb": [], "adjective": [], "adverb": []}
    for w in filtered:
        pos = infer_pos(w)
        pos_buckets.setdefault(pos, []).append(w)

    quotas = {"noun": 1050, "verb": 500, "adjective": 400, "adverb": 50}
    selected: list[str] = []
    selected_set: set[str] = set()
    pos_counts = {"noun": 0, "verb": 0, "adjective": 0, "adverb": 0}

    for pos, quota in quotas.items():
        for w in pos_buckets.get(pos, []):
            if pos_counts[pos] >= quota:
                break
            if not has_specific_meaning(w):
                continue
            if w not in selected_set:
                selected.append(w)
                selected_set.add(w)
                pos_counts[pos] += 1

    for w in filtered:
        if len(selected) >= TARGET_SIZE:
            break
        if not has_specific_meaning(w):
            continue
        if w not in selected_set:
            selected.append(w)
            selected_set.add(w)

    if len(selected) < TARGET_SIZE:
        for w in filtered:
            if len(selected) >= TARGET_SIZE:
                break
            if w not in selected_set:
                selected.append(w)
                selected_set.add(w)

    selected = selected[:TARGET_SIZE]

    df = build_dataset(selected)
    csv_path = DATA_DIR / "engineering_vocab_2000.csv"
    json_path = DATA_DIR / "engineering_vocab_2000.json"
    md_path = DATA_DIR / "engineering_vocab_2000_table.md"
    added_path = DATA_DIR / "added_advanced_words_list.csv"
    quality_path = DATA_DIR / "quality_check_2000.csv"
    quality_summary_path = DATA_DIR / "quality_summary.txt"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| English Word | IPA | Part of Speech | 日本語の意味 | Example Sentence |\n")
        f.write("|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(
                f"| {r['word']} | {r['ipa']} | {r['pos']} | {r['meaning_ja']} | {r['example_en']} |\\n"
            )

    added_df = pd.DataFrame(
        sorted(
            [{"word": w, "pos": p} for w, p in ADVANCED_EXAM_WORDS.items()],
            key=lambda x: (x["pos"], x["word"]),
        )
    )
    added_df.to_csv(added_path, index=False, encoding="utf-8-sig")

    quality_df = quality_check(df)
    quality_df.to_csv(quality_path, index=False, encoding="utf-8-sig")
    success_rate = float(quality_df["overall_ok"].mean())
    with quality_summary_path.open("w", encoding="utf-8") as f:
        f.write(f"overall_success_rate={success_rate:.4f}\n")
        f.write(f"passed={int(quality_df['overall_ok'].sum())}\n")
        f.write(f"total={len(quality_df)}\n")

    print(f"Generated: {len(df)} words")
    print(csv_path)
    print(json_path)
    print(md_path)
    print(added_path)
    print(quality_path)
    print(quality_summary_path)


if __name__ == "__main__":
    main()
