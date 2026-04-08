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
    "mistress", "nervous", "newspaper", "nonviolent", "olecranon", "odontoid",
    "army", "boy", "hornbeam",
    "steelers", "scientology", "supermodel", "powerpoint", "willpower",
    "firepower", "superpower", "powerless", "psychoanalysis",
    "malloy", "ultron", "valverde", "micronesia", "megatron", "aerosmith",
    "sunbeam", "steelhead", "machina", "powershell", "powerball", "powerlifting",
    "headmistress", "truckload", "redesignated", "powertrain", "powerlessness",
    "shitload", "discography", "historiography", "hallucinogenic",
    "pathogenic", "transgenic", "photogenic", "circuitous",
    "superpowers", "uncontrolled", "uncontrollable", "indestructible",
    "unstructured", "unrestrained", "unobstructed", "unprocessed",
    "unconstrained", "unstressed", "reconstructive",
    # Non-engineering biological taxonomy / proper nouns / cultural
    "boatload", "busload", "cartload", "caseload", "shipload", "trainload",
    "moonbeam", "crossbeam", "beamish", "beamy",
    "brainpower", "candlepower", "powerboat", "powerbroker", "powerfulness", "powerplay",
    "songstress", "sempstress", "postmistress", "schoolmistress", "headmistressship",
    "ancestress", "taskmistress",
    "citron", "omicron", "demeter", "faeroes", "faeroese", "fabrice", "chaeronea",
    "columnea", "pumpernickel", "pumpkinseed", "wampumpeag",
    "loadstar", "loadstone", "freeloader", "breechloader", "processional",
    "fogsignal", "signalnoise",
    # Biological / zoological / botanical taxonomy (not academic engineering)
    "micrococcaceae", "micrococcus", "microdesmidae", "microdipodops",
    "micropogonias", "micropterus", "micromeria", "microhylidae",
    "microgametophyte", "microgramma", "micromyx", "micronase",
    "microsporidian", "microsporophyll", "microsporum", "microstomus",
    "microstrobos", "microtus", "microzide", "microsorium", "microbrachia",
    "microcephalus", "microcephaly", "microchiroptera", "microcyte", "microcytosis",
    "micropyle", "microgliacyte", "microsd",
    "nanomia", "nanophthalmos", "nanocephaly",
    "sphaerocarpaceae", "sphaerobolaceae", "sphaerocarpales",
    "sphaerocarpos", "sphaerocarpus",
    "cephalochordata", "urochordata", "chordata", "caudata",
    "rachycentron", "ramphomicron", "anastatica", "aerobacter",
    "acousticophobia", "statice",
    # Obscure paleo / medical  duplicates (British spelling)
    "palaeoanthropology", "palaeobiology", "palaeoclimatology", "palaeodendrology",
    "palaeoecology", "palaeoethnography", "palaeogeography", "palaeogeology",
    "palaeology", "palaeopathology", "palaeornithology", "palaeozoology", "palaetiology",
    "paleoanthropology", "paleobiology", "paleoclimatology", "paleodendrology",
    "paleoecology", "paleoethnography", "paleogeography", "paleopathology",
    "paleornithology", "paleozoology", "paletiology",
    # Obscure medical terms
    "acantholysis", "cystoparalysis", "onycholysis", "aerodontalgia",
    "thermalgesia", "micrometeoritic", "steelyard",
    # British spelling duplicates
    "foetology", "foetometry", "foetoscope", "oesophagoscope",
    "haematolysis",
    # Non-engineering brand names / proper nouns / slang / misspellings
    "aeroflot", "aeron", "computerworld", "indesign", "irobot",
    "powerade", "powerfull", "powerpuff", "powerpc", "powershot", "powerup",
    "vanderpump", "warframe", "megaupload", "machinima", "mechanicsburg",
    "fabricius", "modelo", "modell", "freeloaders", "freeloading",
    "sunbeams", "steeler", "truckloads", "caseloads",
    "soundsystem", "reupload", "uploader", "downloader",
    "loadout", "framerate", "flowy", "fluidly", "frameless",
    "machinegun", "microrna", "micrornas", "powerpuff",
    "airpower", "bootloader", "powerline", "tensorflow",
    # Final cleanup batch
    "boatloads", "machineguns", "mechanicsville", "micronesian",
    "nanowrimo", "powerbomb", "powerbook", "powerlifter", "powerlines",
    "robotnik", "stackoverflow", "trussell", "gamechanger",
    "microtransactions", "microblogging", "systemwide",
    "firstenergy", "powerlifters", "pumpin", "steelbook",
    "aeropostale",
    # Non-engineering words producing katakana-only
    "pumpkin", "pumpkins", "pumpkin-shaped",
    "pumper", "pumped-up",
    "horoscope", "seamstress", "seamstresses",
    "beaman", "beamer", "steele", "steely",
    "patron", "matron", "patron",
    "mistresses", "metatron", "textron", "voltron",
    "jumbotron", "cosmotron", "cybertron", "tron",
    "supermodels", "superpowered",
    "doxology", "eschatology", "teleology", "theology",
    "astrology", "horoscope",
    "calligraphy", "choreography", "autobiography",
    "mythology", "anthology", "bibliography", "biography",
    "amphibology", "phraseology", "doxology",
    "penology", "posology", "psephology", "sinology",
    "cosmetology", "criminology", "musicology", "futurology",
    "epistemology", "phenomenology", "philology", "ideology",
    "gerontology", "ichthyology", "ornithology", "parasitology",
    "ethology", "limnology", "proctology", "odontology",
    "entomology", "epidemiology", "embryology",
    "rheumatology", "oenology", "osteology", "myology",
    "endocrinology", "anesthesiology", "kaleidoscope",
    "hexameter", "octameter", "pentameter",
    "kinetoscope", "kinescope", "stereoscope", "stethoscope",
    "laparoscope", "otoscope", "periscope",
    "ecstatic", "prostatic", "metastatic",
    "optimism", "optimist", "optimists", "optimistically",
    "dissimulation", "materialism", "materialistic", "materialists",
    "immaterial", "materialisation", "materialization",
    "materialise",
    "strainer", "stressor", "stressors",
    "structuralism", "obstructionist",
    "obstructive", "instructional", "instructive", "instructress",
    "recurrent", "concurrently",
    "archaeology", "archeology",
    "chronology", "etymology",
    "acupressure", "carcinogenic", "iatrogenic",
    "kinesiology", "anthropology", "anthropometry",
    "anthropogenic", "immunogenic", "monogenic",
    "polygenic", "eugenic", "biogenic",
    "allergenic", "antigenic", "oncogenic",
    "iconography", "lexicography",
    "professionalism",
    "digitalisation", "digitalis", "datable",
    # Useless words / Not for engineering quiz
    "high-flown", "flown", "remodel", "remodeled", "remodeling",
    "remodelled", "remodelling", "procession", "processions",
    "machinations", "semiconscious",
    "weldon", "steele",
    # ── Vague-producing non-engineering words (batch 3) ──
    "gimignano", "fabrica", "fabricio", "casteel", "catron",
    "celestron", "galvatron", "comixology", "nanoha", "nanook",
    "elektron", "astron", "petron", "scantron", "natron",
    "datang", "flowin", "npower", "sunpower", "nutrisystem",
    "synology", "systema", "systeme", "steelseries", "roboto",
    "robotech", "poweredge", "powerpoints", "powertrains", "powerups",
    "pumphrey", "pumpers", "moonbeams", "microusb", "microstrategy",
    "microsite", "controll", "iframe", "iframes", "spokesmodel",
    "webdesign", "genderfluid", "bugology", "fossilology", "misology",
    "mixology", "sexology", "reflexology", "garbology", "victimology",
    "angelology", "christology", "martyrology", "hagiology",
    "hagiography", "scatology", "parisology", "demonology",
    "aerophilately", "aerophilatelic", "aerophile",
    "cinemascope", "downloaders", "redownload",
    "disempowered", "disempowering",
    "baseload", "buttload",
    "aeroplan", "aerojet",
    "beamforming", "beamline",
    "keyframe", "keyframes",
    "networker", "networkers",
    "microbrew", "microaggressions", "microdermabrasion",
    "microform", "microlight", "microcentrum",
    "nanorods", "nanocrystalline",
    "tautology", "ethnomusicology", "dialectology",
    "redesignation", "undesignated",
    "destress", "signalized",
    # Batch 5 – remaining vague non-engineering
    "steelcase", "systemoutprintln", "duology", "mistresss", "thermaltake",
    "aeroporto", "destructoid", "disempowerment", "pumpkinhead", "malloys",
    "shakeology", "busloads", "controle", "stron", "microblog",
    "shinano", "eurocontrol", "reuploaded", "mechanica", "stmicroelectronics",
    "welden", "aerodactyl", "computex", "metron",
    # Batch 5 – remaining katakana non-engineering
    "columnists", "datas", "energys", "micromax", "micronor",
    "microsofts", "nanos",
    # Batch 6 – vague non-engineering words
    "jontron", "teradata", "appdata", "arbitron", "digitalocean",
    "aeropress", "beame", "dataran", "wikidata", "powerstroke",
    "powerbank", "powerwall", "cryptozoology", "microtech",
    "microcap", "disempower", "decurrent",
    # Batch 7 – final vague non-engineering
    "escaflowne", "cintron", "destructo", "ondansetron",
    "powerscourt", "ufology", "comedogenic", "andrology",
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
    "compute": "verb", "fabricate": "verb", "redesign": "verb",
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

# ─────────────────────────────────────────
# Comprehensive Japanese translations
# ─────────────────────────────────────────
BASE_JA_MEANINGS = {
    # ── Verbs ──────────────────────────────
    "accelerate": "加速する",
    "accommodate": "収容する",
    "accumulate": "蓄積する",
    "activate": "活性化する",
    "adapt": "適応する",
    "adhere": "付着する",
    "aggregate": "集約する",
    "allocate": "割り当てる",
    "amplify": "増幅する",
    "anticipate": "予期する",
    "approximate": "近似する",
    "assemble": "組み立てる",
    "attenuate": "減衰させる",
    "automate": "自動化する",
    "calibrate": "校正する",
    "cohere": "結合する",
    "collide": "衝突する",
    "compensate": "補償する",
    "compile": "コンパイルする",
    "compress": "圧縮する",
    "compute": "計算する",
    "concentrate": "集中させる",
    "condense": "凝縮する",
    "conduct": "伝導する",
    "configure": "構成する",
    "consolidate": "統合する",
    "constrain": "制約する",
    "construct": "構築する",
    "consume": "消費する",
    "contaminate": "汚染する",
    "converge": "収束する",
    "convert": "変換する",
    "correlate": "相関させる",
    "crystallize": "結晶化する",
    "curtail": "削減する",
    "decelerate": "減速する",
    "decompose": "分解する",
    "deform": "変形させる",
    "degrade": "劣化させる",
    "delineate": "描出する",
    "demonstrate": "実証する",
    "denote": "表す",
    "derive": "導出する",
    "designate": "指定する",
    "detect": "検出する",
    "deviate": "逸脱する",
    "diffuse": "拡散する",
    "diminish": "減少させる",
    "disperse": "分散させる",
    "dissipate": "散逸する",
    "distort": "歪める",
    "diverge": "発散する",
    "eliminate": "除去する",
    "embed": "埋め込む",
    "emit": "放出する",
    "enclose": "囲む",
    "enhance": "高める",
    "enumerate": "列挙する",
    "equate": "等置する",
    "evaporate": "蒸発する",
    "evolve": "進化する",
    "exceed": "超過する",
    "exclude": "除外する",
    "expand": "拡張する",
    "exploit": "活用する",
    "expose": "曝露する",
    "fabricate": "製作する",
    "facilitate": "促進する",
    "fluctuate": "変動する",
    "formulate": "定式化する",
    "fuse": "融合する",
    "generate": "生成する",
    "govern": "支配する",
    "homogenize": "均質化する",
    "hypothesize": "仮説を立てる",
    "illuminate": "照射する",
    "immerse": "浸す",
    "implement": "実装する",
    "induce": "誘起する",
    "infer": "推論する",
    "inhibit": "抑制する",
    "inject": "注入する",
    "insert": "挿入する",
    "integrate": "統合する",
    "interfere": "干渉する",
    "interpolate": "補間する",
    "invert": "反転する",
    "isolate": "分離する",
    "justify": "正当化する",
    "maintain": "維持する",
    "manipulate": "操作する",
    "maximize": "最大化する",
    "mediate": "媒介する",
    "migrate": "移行する",
    "minimize": "最小化する",
    "mitigate": "緩和する",
    "modify": "修正する",
    "monitor": "監視する",
    "neutralize": "中和する",
    "normalize": "正規化する",
    "notify": "通知する",
    "optimize": "最適化する",
    "oscillate": "振動する",
    "partition": "分割する",
    "penetrate": "浸透する",
    "perceive": "知覚する",
    "persist": "持続する",
    "perturb": "摂動を与える",
    "postulate": "仮定する",
    "predict": "予測する",
    "preserve": "保持する",
    "process": "処理する",
    "propagate": "伝搬する",
    "quantify": "定量化する",
    "radiate": "放射する",
    "randomize": "無作為化する",
    "react": "反応する",
    "reconcile": "調整する",
    "reconstruct": "再構築する",
    "redistribute": "再分配する",
    "redesign": "再設計する",
    "refine": "精製する",
    "regulate": "制御する",
    "reinforce": "補強する",
    "replicate": "再現する",
    "resolve": "解決する",
    "restrain": "拘束する",
    "restructure": "再構成する",
    "retain": "保持する",
    "retrieve": "取得する",
    "reverse": "逆転させる",
    "rotate": "回転する",
    "saturate": "飽和させる",
    "simulate": "模擬する",
    "specify": "指定する",
    "stabilize": "安定化する",
    "stimulate": "刺激する",
    "substitute": "代替する",
    "suppress": "抑制する",
    "sustain": "維持する",
    "synthesize": "合成する",
    "transform": "変換する",
    "transmit": "伝達する",
    "truncate": "切り捨てる",
    "validate": "妥当性を確認する",
    "verify": "検証する",
    "withstand": "耐える",
    "obstruct": "妨害する",
    "instruct": "指示する",
    "destruct": "破壊する",
    "materialize": "具現化する",
    "empower": "権限を与える",
    "reload": "再読み込みする",
    "unload": "荷卸しする",
    "download": "ダウンロードする",
    "upload": "アップロードする",
    # ── Nouns (with commonly wrong stem matches) ──
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
    "modulus": "弾性率",
    "optimization": "最適化",
    "oscillation": "振動",
    "permeability": "透過率",
    "plasticity": "塑性",
    "porosity": "空隙率",
    "resilience": "復元性",
    "shear": "せん断",
    "stiffness": "剛性",
    "torsion": "ねじり",
    "viscosity": "粘性",
    # Nouns that get WRONG stem translations:
    "construction": "建設",
    "infrastructure": "社会基盤",
    "instruction": "命令",
    "instructions": "命令",
    "destruction": "破壊",
    "instructor": "指導者",
    "reconstruction": "再建",
    "obstruction": "妨害",
    "designation": "指定",
    "designer": "設計者",
    "restraint": "拘束",
    "constraint": "制約",
    "constraints": "制約条件",
    "distress": "苦痛",
    "empowerment": "権限付与",
    "overload": "過負荷",
    "loading": "負荷",
    "processor": "処理装置",
    "processing": "処理",
    "mechanism": "機構",
    "manufacturing": "製造",
    "constructiveness": "建設的性質",
    "indestructibility": "不壊性",
    "deconstructionism": "脱構築主義",
    "restructuring": "再構成",
    "instrumentation": "計装",
    "self-control": "自制",
    "self-analysis": "自己分析",
    "self-destruction": "自壊",
    # Engineering nouns
    "thermodynamics": "熱力学",
    "thermochemistry": "熱化学",
    "thermoelasticity": "熱弾性",
    "electrochemistry": "電気化学",
    "electrodynamics": "電気力学",
    "electromagnetics": "電磁気学",
    "electromechanics": "電気機械工学",
    "electrostatics": "静電気学",
    "microstructure": "微細構造",
    "macrostructure": "マクロ構造",
    "metallurgy": "冶金学",
    "tribology": "トライボロジー",
    "rheology": "レオロジー",
    "continuum": "連続体",
    "continuity": "連続性",
    "compressibility": "圧縮率",
    "permittivity": "誘電率",
    "capacitance": "静電容量",
    "inductance": "インダクタンス",
    "susceptibility": "感受率",
    "admittance": "アドミタンス",
    "reactance": "リアクタンス",
    "resistivity": "抵抗率",
    "conductance": "コンダクタンス",
    "diffusivity": "拡散率",
    "transmissivity": "透過率",
    "reflectivity": "反射率",
    "absorptivity": "吸収率",
    "emissivity": "放射率",
    "crystallization": "結晶化",
    "solidification": "凝固",
    "vaporization": "気化",
    "condensation": "凝縮",
    "nucleation": "核生成",
    "granularity": "粒度",
    "anisotropy": "異方性",
    "isotropy": "等方性",
    "viscoelasticity": "粘弾性",
    "buckling": "座屈",
    "fracture": "破壊",
    "sintering": "焼結",
    "annealing": "焼鈍",
    "quenching": "焼入れ",
    "tempering": "焼戻し",
    "scattering": "散乱",
    "interference": "干渉",
    "diffraction": "回折",
    "polarization": "偏光",
    "modulation": "変調",
    "demodulation": "復調",
    "multiplexing": "多重化",
    "sampling": "サンプリング",
    "quantization": "量子化",
    "linearization": "線形化",
    "stabilization": "安定化",
    "regularization": "正則化",
    "discretization": "離散化",
    "normalization": "正規化",
    "parameterization": "パラメータ化",
    "synchronization": "同期化",
    "parallelization": "並列化",
    "vectorization": "ベクトル化",
    "serialization": "直列化",
    "scalability": "拡張性",
    "reliability": "信頼性",
    "availability": "可用性",
    "maintainability": "保守性",
    "traceability": "追跡可能性",
    "repeatability": "再現性",
    "reproducibility": "再現性",
    "calibration": "校正",
    "validation": "妥当性検証",
    "verification": "検証",
    "benchmarking": "ベンチマーク",
    "actuation": "駆動",
    "localization": "局在化",
    "navigation": "航法",
    "trajectory": "軌道",
    "kinematics": "運動学",
    "dynamics": "動力学",
    "statics": "静力学",
    "hydrodynamics": "流体力学",
    "aerodynamics": "空気力学",
    "biomechanics": "生体力学",
    "geomechanics": "地盤力学",
    "seismology": "地震学",
    "hydraulics": "水力学",
    "pneumatics": "空気圧工学",
    "eigenvalue": "固有値",
    "eigenvector": "固有ベクトル",
    "covariance": "共分散",
    "correlation": "相関",
    "regression": "回帰",
    "approximation": "近似",
    "interpolation": "補間",
    "extrapolation": "外挿",
    "perturbation": "摂動",
    "convergence": "収束",
    "divergence": "発散",
    "stability": "安定性",
    "instability": "不安定性",
    "resonance": "共振",
    "damping": "減衰",
    "hardening": "硬化",
    "softening": "軟化",
    "creep": "クリープ",
    "wear": "摩耗",
    "corrosion": "腐食",
    "oxidation": "酸化",
    "reduction": "還元",
    "catalysis": "触媒作用",
    "adsorption": "吸着",
    "filtration": "濾過",
    "sedimentation": "沈殿",
    "distillation": "蒸留",
    "extraction": "抽出",
    "purification": "精製",
    "polymerization": "重合",
    "semiconductor": "半導体",
    "transistor": "トランジスタ",
    "bandwidth": "帯域幅",
    "latency": "遅延",
    "throughput": "スループット",
    "protocol": "プロトコル",
    "cryptography": "暗号学",
    "encryption": "暗号化",
    "decryption": "復号",
    "computation": "計算",
    "probability": "確率",
    "heuristics": "ヒューリスティクス",
    "dielectric": "誘電体",
    "firmware": "ファームウェア",
    "middleware": "ミドルウェア",
    "redundancy": "冗長性",
    "robustness": "頑健性",
    "spectrum": "スペクトル",
    "topology": "位相幾何学",
    "manifold": "多様体",
    "equilibrium": "平衡",
    "gradient": "勾配",
    "momentum": "運動量",
    "parameter": "パラメータ",
    "hypothesis": "仮説",
    "criterion": "基準",
    "regime": "領域",
    "threshold": "閾値",
    "analogy": "類推",
    "anomaly": "異常",
    "incentive": "誘因",
    "notion": "概念",
    "nuance": "微妙な差異",
    "precedent": "先例",
    "variant": "変異体",
    # Additional nouns
    "microwave": "マイクロ波",
    "ecosystem": "生態系",
    "psychology": "心理学",
    "photography": "写真術",
    "sociology": "社会学",
    "nano": "ナノ",
    "ceramics": "セラミックス",
    "physiology": "生理学",
    "overflow": "オーバーフロー",
    "pathology": "病理学",
    "metadata": "メタデータ",
    "microscopy": "顕微鏡法",
    "microbiology": "微生物学",
    "biotechnology": "バイオテクノロジー",
    "gastroenterology": "消化器病学",
    "electromyography": "筋電図法",
    "otorhinolaryngology": "耳鼻咽喉科学",
    # More technical nouns
    "technology": "技術",
    "database": "データベース",
    "framework": "枠組み",
    "robotics": "ロボット工学",
    "acoustics": "音響学",
    "circuitry": "回路",
    "dataset": "データセット",
    "mainframe": "メインフレーム",
    "timeframe": "時間枠",
    "workflow": "ワークフロー",
    "airflow": "気流",
    "outflow": "流出",
    "inflow": "流入",
    "payload": "ペイロード",
    "workload": "作業負荷",
    "horsepower": "馬力",
    "manpower": "人的資源",
    "powerhouse": "発電所",
    "nanoparticles": "ナノ粒子",
    "nanoparticle": "ナノ粒子",
    "microbes": "微生物",
    "microorganism": "微生物",
    "microorganisms": "微生物",
    "neurology": "神経学",
    "radiology": "放射線学",
    "antimicrobial": "抗菌の",
    "biomedical": "生体医工学の",
    "stressful": "応力の多い",
    "biomedical": "生体医工学の",
    "columns": "柱",
    "column": "柱",
    "welding": "溶接",
    "weldability": "溶接性",
    "polymer": "高分子",
    "ceramic": "セラミック",
    "composite": "複合材",
    "turbine": "タービン",
    "transformer": "変圧器",
    "amplifier": "増幅器",
    "oscillator": "発振器",
    "transducer": "変換器",
    "actuator": "アクチュエータ",
    "substrate": "基板",
    "electrode": "電極",
    "catalyst": "触媒",
    "alloy": "合金",
    "bearing": "軸受",
    "cantilever": "片持ち梁",
    "cavity": "空洞",
    "compliance": "コンプライアンス",
    "curvature": "曲率",
    "deflection": "たわみ",
    "dimension": "寸法",
    "elongation": "伸び",
    "entropy": "エントロピー",
    "hardness": "硬度",
    "lattice": "格子",
    "moment": "モーメント",
    "precision": "精度",
    "prototype": "試作品",
    "rigidity": "剛性",
    "tolerance": "公差",
    "yield": "降伏",
    "zonation": "帯状分布",
    "copolymer": "共重合体",
    "photodiode": "フォトダイオード",
    "optoelectronics": "光エレクトロニクス",
    "microcontroller": "マイクロコントローラ",
    "checksum": "チェックサム",
    "bandgap": "バンドギャップ",
    "servo": "サーボ",
    "feedforward": "フィードフォワード",
    "feedback": "フィードバック",
    "desorption": "脱着",
    "centrifugation": "遠心分離",
    "etching": "エッチング",
    "deposition": "堆積",
    "cad": "CAD（コンピュータ支援設計）",
    "cam": "CAM（コンピュータ支援製造）",
    "truss": "トラス",
    "beam": "梁",
    "load": "荷重",
    "stress": "応力",
    "strain": "ひずみ",
    "voltage": "電圧",
    "current": "電流",
    "resistance": "抵抗",
    "concrete": "コンクリート",
    "steel": "鋼",
    "signal": "信号",
    "sensor": "センサ",
    "control": "制御",
    "design": "設計",
    "system": "システム",
    "model": "モデル",
    "network": "ネットワーク",
    "data": "データ",
    "computer": "コンピュータ",
    "machine": "機械",
    "frame": "フレーム",
    "phase": "位相",
    "efficiency": "効率",
    "pump": "ポンプ",
    "valve": "バルブ",
    "flow": "流れ",
    "power": "電力",
    "energy": "エネルギー",
    "pressure": "圧力",
    "battery": "電池",
    "robot": "ロボット",
    "foundation": "基礎",
    "material": "材料",
    "structure": "構造",
    "process": "処理する",
    "simulation": "シミュレーション",
    "vibration": "振動",
    "manufacturing": "製造",
    "mechanism": "機構",
    "assembly": "組立",
    "attenuation": "減衰",
    "transience": "過渡性",
    "flexural": "曲げの",
    "torsional": "ねじりの",
    "tensility": "引張性",
    "orthotropic": "直交異方性の",
    "orthotropy": "直交異方性",
    "elastoplasticity": "弾塑性",
    "incompressibility": "非圧縮性",
    "fractography": "破面解析",
    "microfabrication": "微細加工",
    "nanofabrication": "ナノ加工",
    "photolithography": "フォトリソグラフィー",
    "machinability": "被削性",
    "formability": "成形性",
    "printability": "印刷適性",
    "identifiability": "同定可能性",
    "observability": "可観測性",
    "controllability": "可制御性",
    "desynchronization": "非同期化",
    "interoperability": "相互運用性",
    "thermofluid": "熱流体",
    "geostatics": "地盤静力学",
    "transconductance": "相互コンダクタンス",
    "determinism": "決定論",
    "algorithmic": "アルゴリズムの",
    "plasticization": "可塑化",
    "permeance": "パーミアンス",
    "refractivity": "屈折率",
    # More technical nouns and compounds
    "spectrometry": "分光分析",
    "chromatography": "クロマトグラフィー",
    "hydrolysis": "加水分解",
    "machinist": "機械工",
    "filesystem": "ファイルシステム",
    "datacenter": "データセンター",
    "flowchart": "フローチャート",
    "systematics": "分類学",
    "aerodrome": "飛行場",
    "homology": "相同性",
    "urology": "泌尿器科学",
    "airframe": "機体",
    "undercurrent": "底流",
    "analogue": "アナログ",
    "analogous": "類似の",
    "foundational": "基礎的な",
    "vibrational": "振動の",
    "polymeric": "高分子の",
    "isothermal": "等温の",
    "isobaric": "等圧の",
    "isochoric": "等積の",
    "adiabatic": "断熱の",
    "endothermic": "吸熱の",
    "exothermic": "発熱の",
    "stoichiometric": "化学量論の",
    "calorimetry": "熱量測定",
    "gravimetry": "重量分析",
    "spectroscopy": "分光法",
    "rheological": "レオロジーの",
    "cryogenic": "極低温の",
    "thermoplastic": "熱可塑性の",
    "thermoset": "熱硬化性の",
    "piezoelectric": "圧電の",
    "ferromagnetic": "強磁性の",
    "paramagnetic": "常磁性の",
    "diamagnetic": "反磁性の",
    "superconductor": "超伝導体",
    "superconducting": "超伝導の",
    "photovoltaic": "太陽光発電の",
    "microelectronics": "マイクロエレクトロニクス",
    "nanostructure": "ナノ構造",
    "nanotechnology": "ナノテクノロジー",
    "superconductivity": "超伝導性",
    "microprocessor": "マイクロプロセッサ",
    "microsensor": "マイクロセンサ",
    "biosensor": "バイオセンサ",
    "bioreactor": "バイオリアクター",
    "biocompatible": "生体適合性の",
    "geothermal": "地熱の",
    "geotechnical": "地盤工学の",
    "hydrothermal": "水熱の",
    "electrochemical": "電気化学の",
    "electromechanical": "電気機械の",
    "electrostatic": "静電の",
    "thermodynamic": "熱力学的な",
    "aerodynamic": "空気力学の",
    "hydrodynamic": "流体力学の",
    "biomechanical": "生体力学の",
    "photochemical": "光化学の",
    "stoichiometry": "化学量論",
    "potentiometer": "電位差計",
    "accelerometer": "加速度計",
    "interferometer": "干渉計",
    "voltmeter": "電圧計",
    "ammeter": "電流計",
    "galvanometer": "検流計",
    "barometer": "気圧計",
    "thermometer": "温度計",
    "hygrometer": "湿度計",
    "manometer": "圧力計",
    "dynamometer": "動力計",
    "oscilloscope": "オシロスコープ",
    "spectrometer": "分光器",
    "micrometer": "マイクロメータ",
    "viscometer": "粘度計",
    "flowmeter": "流量計",
    "tachometer": "回転速度計",
    "magnetometer": "磁力計",
    "pyrometer": "高温計",
    "telemetry": "テレメトリー",
    "biometrics": "生体認証",
    "supercomputer": "スーパーコンピュータ",
    "subsystem": "サブシステム",
    "substation": "変電所",
    "hydropower": "水力発電",
    "typographer": "タイポグラファー",
    "typography": "タイポグラフィー",
    "toxicology": "毒物学",
    "immunology": "免疫学",
    "oncology": "腫瘍学",
    "dermatology": "皮膚科学",
    "cardiology": "心臓学",
    "ophthalmology": "眼科学",
    "pharmacology": "薬理学",
    "paleontology": "古生物学",
    "meteorology": "気象学",
    "mineralogy": "鉱物学",
    "volcanology": "火山学",
    "petrology": "岩石学",
    "oceanography": "海洋学",
    "topography": "地形学",
    "cartography": "地図学",
    "lithography": "リソグラフィー",
    "holography": "ホログラフィー",
    "tomography": "断層撮影法",
    "densitometry": "濃度測定法",
    "photometry": "測光法",
    "colorimetry": "比色法",
    "respirometry": "呼吸測定法",
    "psychrometry": "湿度測定法",
    "dosimetry": "線量測定法",
    "audiometry": "聴力測定法",
    "tribometer": "摩擦計",
    "microfluidics": "マイクロ流体工学",
    "mechatronics": "メカトロニクス",
    "avionics": "航空電子工学",
    "photonics": "フォトニクス",
    "plasmonics": "プラズモニクス",
    # ── -ology words ──────────────────────────
    "phonology": "音韻学",
    "typology": "類型学",
    "hematology": "血液学",
    "haematology": "血液学",
    "gynaecology": "婦人科学",
    "bacteriology": "細菌学",
    "cytology": "細胞学",
    "enzymology": "酵素学",
    "mycology": "菌類学",
    "enology": "醸造学",
    "conchology": "貝類学",
    "craniology": "頭蓋学",
    "herpetology": "爬虫類学",
    "lexicology": "語彙学",
    "lithology": "岩石学",
    "selenology": "月学",
    "horology": "時計学",
    "pedology": "土壌学",
    "phycology": "藻類学",
    "phytology": "植物学",
    "algology": "藻類学",
    "teratology": "奇形学",
    "thanatology": "死生学",
    "speleology": "洞窟学",
    "spelaeology": "洞窟学",
    "necrology": "死亡記録",
    "neology": "新語学",
    "tocology": "産科学",
    "zymology": "発酵学",
    "primatology": "霊長類学",
    "sociobiology": "社会生物学",
    "exobiology": "宇宙生物学",
    "radiobiology": "放射線生物学",
    "agrobiology": "農業生物学",
    "astrobiology": "宇宙生物学",
    "morphophysiology": "形態生理学",
    "psychophysiology": "精神生理学",
    "psychopharmacology": "精神薬理学",
    "neonatology": "新生児学",
    "perinatology": "周産期学",
    "nephrology": "腎臓学",
    "otology": "耳科学",
    "ecclesiology": "教会学",
    "axiology": "価値論",
    "symbology": "記号学",
    "escapology": "脱出術",
    "stemmatology": "系統学",
    "traumatology": "外傷学",
    "rhinolaryngology": "鼻咽喉科学",
    "immunopathology": "免疫病理学",
    "chemoimmunology": "化学免疫学",
    "glottochronology": "語彙統計年代学",
    "lepidopterology": "鱗翅目学",
    "semasiology": "意味論",
    "soteriology": "救済論",
    "liturgiology": "典礼学",
    "patrology": "教父学",
    "poenology": "刑罰学",
    "malacology": "軟体動物学",
    "pomology": "果実学",
    "heterology": "異質学",
    "sumerology": "シュメール学",
    "assyriology": "アッシリア学",
    "agrology": "農学",
    "numismatology": "貨幣学",
    "pteridology": "シダ植物学",
    "allergology": "アレルギー学",
    "angiology": "血管学",
    "fetology": "胎児学",
    "nephology": "雲学",
    "oology": "卵学",
    "psychopathology": "精神病理学",
    "pathophysiology": "病態生理学",
    "aetiology": "病因学",
    "numerology": "数秘術",
    "metrology": "計量学",
    "crystallography": "結晶学",
    "microeconomics": "ミクロ経済学",
    "biosystematy": "生物分類学",
    "ology": "学問",
    # ── -graphy words ──────────────────────────
    "seismography": "地震記録法",
    "xerography": "ゼログラフィー",
    "serigraphy": "シルクスクリーン法",
    "echocardiography": "心エコー法",
    "roentgenography": "レントゲン撮影",
    "venography": "静脈造影",
    "myelography": "脊髄造影",
    "chirography": "書道",
    "cacography": "悪筆",
    "ideography": "表意文字法",
    "anemography": "風力記録",
    "hypsography": "等高線法",
    "lymphangiography": "リンパ管造影",
    "pasigraphy": "万国文字",
    "skiagraphy": "X線撮影",
    "adoxography": "平凡な主題の文章",
    "glyptography": "宝石彫刻術",
    "cholangiography": "胆管造影",
    "orography": "山岳学",
    "orology": "山岳学",
    "chromolithography": "多色石版術",
    "radiophotography": "放射線写真撮影",
    "radiotelegraphy": "無線電信",
    "anorthography": "書字障害",
    "mammothermography": "乳房温度撮影",
    "xeroradiography": "ゼロラジオグラフィー",
    # ── -metry/-meter words ──────────────────────────
    "acidimetry": "酸度測定",
    "actinometry": "放射測定",
    "algometry": "痛覚測定",
    "allometry": "相対成長",
    "bathymetry": "水深測量",
    "cephalometry": "頭部計測",
    "pleximetry": "打診",
    "pelvimetry": "骨盤計測",
    "fetometry": "胎児計測",
    "hypsometry": "標高測定",
    "spirometry": "肺活量測定",
    "tonometry": "眼圧測定",
    "viscosimetry": "粘度測定",
    "cytophotometry": "細胞光度測定",
    "planimeter": "プラニメーター",
    "clinometer": "傾斜計",
    "inclinometer": "傾斜計",
    "declinometer": "偏角計",
    "eudiometer": "ガス分析器",
    "evaporometer": "蒸発計",
    "gasometer": "ガスタンク",
    "gaussmeter": "ガウスメーター",
    "goniometer": "測角器",
    "gravimeter": "重力計",
    "heliometer": "太陽径測定器",
    "oximeter": "酸素飽和度計",
    "psychrometer": "乾湿計",
    "pelvimeter": "骨盤計",
    "tensimeter": "張力計",
    "tensiometer": "表面張力計",
    "polarimeter": "偏光計",
    "salinometer": "塩分計",
    "sensitometer": "感度計",
    "spherometer": "球面計",
    "tacheometer": "タキメーター",
    "udometer": "雨量計",
    "variometer": "バリオメーター",
    "pluviometer": "雨量計",
    "mileometer": "走行距離計",
    "milometer": "走行距離計",
    "densimeter": "密度計",
    "dasymeter": "ガス密度計",
    "katharometer": "カタロメーター",
    "reflectometer": "反射率計",
    "refractometer": "屈折計",
    "rheometer": "レオメーター",
    "algometer": "痛覚計",
    "bolometer": "ボロメーター",
    "bathometer": "水深計",
    "bathymeter": "水深計",
    "tintometer": "色度計",
    "radiomicrometer": "放射マイクロメーター",
    "colorimeter": "比色計",
    "cytophotometer": "細胞光度計",
    "hectometer": "ヘクトメートル",
    "dekameter": "デカメートル",
    "femtometer": "フェムトメートル",
    "picometer": "ピコメートル",
    "myriameter": "ミリアメートル",
    "milliammeter": "ミリアンメーター",
    "millivoltmeter": "ミリボルトメーター",
    "atmometer": "蒸発計",
    "plessimeter": "打診板",
    "pleximeter": "打診板",
    "volumeter": "容積計",
    "viscosimeter": "粘度計",
    "thermogravimeter": "熱重量計",
    "dosimeter": "線量計",
    # ── -scope words ──────────────────────────
    "colonoscope": "大腸内視鏡",
    "culdoscope": "ダグラス窩鏡",
    "epidiascope": "映写機",
    "esophagoscope": "食道鏡",
    "fetoscope": "胎児聴診器",
    "fluoroscope": "透視装置",
    "hodoscope": "粒子検出器",
    "hygroscope": "吸湿計",
    "iconoscope": "アイコノスコープ",
    "kinescope": "キネスコープ",
    "kinetoscope": "キネトスコープ",
    "nephoscope": "雲鏡",
    "ophthalmoscope": "検眼鏡",
    "polariscope": "偏光器",
    "proctoscope": "直腸鏡",
    "sigmoidoscope": "S状結腸鏡",
    "stroboscope": "ストロボスコープ",
    "synchroscope": "同期検定器",
    "synchronoscope": "同期検定器",
    "tachistoscope": "タキストスコープ",
    "auriscope": "耳鏡",
    "auroscope": "耳鏡",
    "roentgenoscope": "レントゲン透視鏡",
    "gyroscope": "ジャイロスコープ",
    # ── -lysis words ──────────────────────────
    "glycolysis": "解糖",
    "hemolysis": "溶血",
    "haemolysis": "溶血",
    "fibrinolysis": "線維素溶解",
    "proteolysis": "タンパク質分解",
    "cytolysis": "細胞溶解",
    "bacteriolysis": "細菌溶解",
    "karyolysis": "核溶解",
    "necrolysis": "壊死性溶解",
    "spasmolysis": "痙攣解除",
    "zymolysis": "発酵",
    "amylolysis": "デンプン分解",
    "cryptanalysis": "暗号解読",
    "hypnoanalysis": "催眠分析",
    "dielectrolysis": "電気分解",
    "uranalysis": "尿分析",
    "lysis": "溶解",
    "hemodialysis": "血液透析",
    "haemodialysis": "血液透析",
    # ── -phase words ──────────────────────────
    "anaphase": "後期",
    "metaphase": "中期",
    "prophase": "前期",
    "telophase": "終期",
    # ── -tron/-tronics words ──────────────────
    "klystron": "クライストロン",
    "magnetron": "マグネトロン",
    "bevatron": "ベバトロン",
    "plastron": "胸当て",
    "plectron": "義甲",
    "intron": "イントロン",
    "elytron": "鞘翅",
    "cistron": "シストロン",
    "mesotron": "中間子",
    "synchrocyclotron": "シンクロサイクロトロン",
    "millimicron": "ミリミクロン",
    "antielectron": "陽電子",
    "antineutron": "反中性子",
    # ── -genic adjectives ──────────────────────────
    "mutagenic": "突然変異原性の",
    "estrogenic": "エストロゲン性の",
    "teratogenic": "催奇形性の",
    "icterogenic": "黄疸原性の",
    "lactogenic": "催乳性の",
    "glycogenic": "グリコーゲン生成の",
    "lysogenic": "溶原性の",
    "cacogenic": "劣性遺伝の",
    "dysgenic": "劣性遺伝の",
    "ketogenic": "ケトン生成の",
    "haematogenic": "血液生成の",
    "collagenic": "コラーゲン性の",
    "anorexigenic": "食欲抑制の",
    "cytopathogenic": "細胞病原性の",
    "exogenic": "外因性の",
    # ── -static adjectives ──────────────────────────
    "bacteriostatic": "静菌の",
    "hemostatic": "止血の",
    "homeostatic": "恒常性の",
    "orthostatic": "起立性の",
    "astatic": "不安定な",
    # ── Other technical adjectives ──────────────────
    "viscoelastic": "粘弾性の",
    "somatosensory": "体性感覚の",
    "sensorineural": "感音性の",
    "extrasensory": "超感覚の",
    "nonconductive": "非導電性の",
    "nonstructural": "非構造の",
    "nonthermal": "非熱的な",
    "nonmaterial": "非物質的な",
    "nonmechanical": "非機械的な",
    "nonmechanistic": "非機械論的な",
    "servomechanical": "サーボ機構の",
    "machinelike": "機械的な",
    "robotlike": "ロボットのような",
    "networklike": "ネットワーク状の",
    "columniform": "円柱状の",
    "columnlike": "円柱状の",
    "endothermal": "吸熱の",
    "exothermal": "発熱の",
    "hyperthermal": "高温の",
    "adynamic": "無力の",
    "aerolitic": "隕石の",
    "aerophilic": "好気性の",
    "aerophilous": "好気性の",
    "anaerobiotic": "嫌気性の",
    "coseismic": "同時地震の",
    "designative": "指定の",
    "distressful": "苦痛な",
    "microcephalous": "小頭症の",
    "micropylar": "珠孔の",
    "microsomal": "ミクロソームの",
    "bivalved": "二枚貝の",
    "noncolumned": "円柱のない",
    "undatable": "年代測定不能の",
    "undesigned": "意図されていない",
    "unmechanical": "機械的でない",
    "unmechanised": "機械化されていない",
    "unmechanized": "機械化されていない",
    "unpowered": "動力のない",
    "unconstructive": "建設的でない",
    "unreconstructed": "再建されていない",
    "unsystematic": "体系的でない",
    "undynamic": "活気のない",
    "unframed": "枠のない",
    "uninstructed": "教育されていない",
    "uninstructive": "教育的でない",
    "unalloyed": "純粋な",
    "powerplant": "発電所",
    # ── Adverbs ──────────────────────────
    "concretely": "具体的に",
    "constructively": "建設的に",
    "currently": "現在",
    "materially": "物質的に",
    "thermally": "熱的に",
    "uncontrollably": "制御不能に",
    "statically": "静的に",
    # ── Other nouns ──────────────────────────
    "servomechanism": "サーボ機構",
    "servosystem": "サーボシステム",
    "minicomputer": "ミニコンピュータ",
    "prefabrication": "プレハブ工法",
    "refabrication": "再製造",
    "aerobe": "好気性生物",
    "anaerobe": "嫌気性生物",
    "aerobiosis": "好気性生存",
    "aeronaut": "気球乗り",
    "aerogramme": "航空書簡",
    "aerophyte": "着生植物",
    "astrodynamics": "天体力学",
    "hemodynamics": "血行力学",
    "psychodynamics": "心理力学",
    "polymerisation": "重合",
    "mechanisation": "機械化",
    "mechanist": "機械論者",
    "modeler": "モデル作成者",
    "modeller": "モデル作成者",
    "phage": "ファージ",
    "bacteriophage": "バクテリオファージ",
    "coliphage": "大腸菌ファージ",
    "mycophage": "菌食者",
    "glucophage": "糖消費菌",
    "microphage": "小食細胞",
    "permalloy": "パーマロイ",
    "flowage": "流動",
    "cashflow": "キャッシュフロー",
    "backflow": "逆流",
    "backflowing": "逆流",
    "fluidness": "流動性",
    "photomicrograph": "顕微鏡写真",
    "compositeness": "合成性",
    "destructibility": "破壊可能性",
    "destructiveness": "破壊性",
    "constructivism": "構成主義",
    "constructivist": "構成主義者",
    "deconstructivism": "脱構築主義",
    "controllership": "管理職",
    "instructorship": "講師職",
    "countercurrent": "逆流",
    "crosscurrent": "横流",
    "occurrent": "発生",
    "currentness": "現在性",
    "noncurrent": "非流動の",
    "immateriality": "非物質性",
    "materiality": "物質性",
    "fluidounce": "液量オンス",
    "fluidram": "液量ドラム",
    "overpressure": "過圧",
    "misconstruction": "誤解",
    "understructure": "下部構造",
    "unconstraint": "無拘束",
    "unrestraint": "無拘束",
    "nonresistance": "無抵抗",
    "weldment": "溶接物",
    "framer": "枠組み製作者",
    "framers": "枠組み製作者",
    "bedframe": "ベッドフレーム",
    "doorframe": "ドア枠",
    "underframe": "下枠",
    "reframe": "再構成",
    "preload": "予荷重",
    "steelworks": "製鉄所",
    "steelmaker": "製鉄業者",
    "steelworker": "製鉄労働者",
    "steelworkers": "製鉄労働者",
    "steelman": "製鉄業者",
    "obstructer": "妨害者",
    "obstructor": "妨害者",
    "restrainer": "拘束器",
    "digitalin": "ジギタリン",
    "digitalisation": "デジタル化",
    "digitalization": "デジタル化",
    "distraint": "差し押さえ",
    "distressfulness": "苦痛性",
    "distressingness": "苦痛性",
    "systematisation": "体系化",
    "systematiser": "体系化者",
    "systematism": "体系主義",
    "systematist": "分類学者",
    "systematization": "体系化",
    "systematizer": "体系化者",
    "systemiser": "体系化者",
    "systemizer": "体系化者",
    "signaler": "信号手",
    "signaller": "信号手",
    "signalisation": "信号化",
    "signalization": "信号化",
    "signalman": "信号手",
    "semiconsciousness": "半意識",
    "columniation": "円柱配列",
    "microfarad": "マイクロファラッド",
    "microfiche": "マイクロフィッシュ",
    "microtaggant": "マイクロタガント",
    "microscopist": "顕微鏡学者",
    "microscopium": "顕微鏡座",
    "microbiota": "微生物叢",
    "microtubule": "微小管",
    "microtubules": "微小管",
    "micrometeoroid": "微小隕石",
    "nanometers": "ナノメートル",
    "micromillimetre": "マイクロミリメートル",
    "overstrain": "過度の緊張",
    "eyestrain": "眼精疲労",
    "waterpower": "水力",
    "designatum": "指示対象",
    "manufactory": "製造所",
    "quercitron": "ケルシトロン",
    "mandatary": "受任者",
    "cabinetwork": "家具製作",
    "concreteness": "具体性",
    "valvelet": "小弁",
    "univalve": "一枚貝",
    "antimicrobic": "抗菌剤",
    "chirology": "手話学",
    # ── Verbs ──────────────────────────
    "offload": "荷を降ろす",
    "offloading": "荷降ろし",
    "overpower": "圧倒する",
    "overpowered": "圧倒された",
    "overpowering": "圧倒的な",
    "computerised": "コンピュータ化された",
    "materialised": "実体化した",
    "mechanised": "機械化された",
    "mechanized": "機械化された",
    "optimised": "最適化された",
    "prefabricated": "プレハブの",
    "reprocessing": "再処理",
    "transistorised": "トランジスタ化された",
    "transistorized": "トランジスタ化された",
    "unstrained": "緊張のない",
    "aerosolised": "エアロゾル化された",
    # ── More remaining vague words ──────────────────
    "aerobatic": "曲技飛行の",
    "hypoallergenic": "低刺激性の",
    "microfluidic": "マイクロ流体の",
    "nondestructive": "非破壊の",
    "psychodynamic": "精神力動の",
    "uncolumned": "円柱のない",
    "analogously": "類似して",
    "destructively": "破壊的に",
    "inefficiently": "非効率的に",
    "microscopically": "顕微鏡的に",
    "astrophotography": "天体写真撮影",
    "cohomology": "コホモロジー",
    "cytometry": "細胞計測",
    "databank": "データバンク",
    "datasheet": "データシート",
    "designator": "指示子",
    "destructor": "デストラクタ",
    "dissymmetry": "非対称性",
    "hepatology": "肝臓学",
    "histopathology": "組織病理学",
    "hodometer": "走行距離計",
    "hypsometer": "測高計",
    "metamaterials": "メタマテリアル",
    "microgauss": "マイクログウス",
    "microsporangium": "小胞子嚢",
    "palaeontology": "古生物学",
    "prestressed": "プレストレストの",
    "reanalysis": "再分析",
    "seismicity": "地震活動",
    "uranology": "天文学",
    "steelmaking": "製鋼",
    "underpowered": "出力不足の",
    "chylomicron": "カイロミクロン",
    "fluidized": "流動化された",
    "machinator": "策略家",
    "machin": "機械",
    "reprocessed": "再処理された",
    "systematized": "体系化された",
    "micromanagement": "マイクロマネジメント",
    "orogenic": "造山運動の",
    "ecstatically": "歓喜して",
    "coprocessor": "コプロセッサ",
    "nanocrystals": "ナノ結晶",
    "preprocessing": "前処理",
    "preprocessor": "プリプロセッサ",
    "remanufactured": "再製造された",
    "seismically": "地震的に",
    "distressingly": "悩ましいほど",
    "streamflow": "河川流量",
    "wireframe": "ワイヤーフレーム",
    "polymerized": "重合された",
    # ── Adjectives ──────────────────────────
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
    "laminar": "層流の",
    "seismic": "地震の",
    "abrupt": "急激な",
    "accurate": "正確な",
    "adjacent": "隣接した",
    "adverse": "不利な",
    "analog": "アナログの",
    "anisotropic": "異方性の",
    "applicable": "適用可能な",
    "arbitrary": "任意の",
    "axial": "軸方向の",
    "balanced": "均衡した",
    "brittle": "脆い",
    "causal": "因果の",
    "cohesive": "凝集性の",
    "coincident": "一致する",
    "coherent": "コヒーレントな",
    "compatible": "互換性のある",
    "compressive": "圧縮の",
    "computational": "計算の",
    "concurrent": "同時発生の",
    "conductive": "導電性の",
    "confined": "閉じ込められた",
    "consistent": "一貫した",
    "constitutive": "構成的な",
    "controllable": "制御可能な",
    "conventional": "従来の",
    "corresponding": "対応する",
    "critical": "臨界の",
    "cumulative": "累積の",
    "dense": "高密度の",
    "deterministic": "決定論的な",
    "differential": "微分の",
    "discrete": "離散的な",
    "distributed": "分布した",
    "dominant": "支配的な",
    "ductile": "延性のある",
    "elastic": "弾性の",
    "empirical": "経験的な",
    "equivalent": "等価の",
    "exponential": "指数関数的な",
    "feasible": "実現可能な",
    "finite": "有限の",
    "frictional": "摩擦の",
    "functional": "機能的な",
    "hybrid": "ハイブリッドの",
    "identical": "同一の",
    "ideal": "理想的な",
    "incremental": "増分の",
    "independent": "独立した",
    "induced": "誘導された",
    "inert": "不活性の",
    "inherent": "固有の",
    "initial": "初期の",
    "integral": "積分の",
    "interactive": "対話的な",
    "intrinsic": "固有の",
    "isotropic": "等方性の",
    "iterative": "反復の",
    "lateral": "横方向の",
    "linear": "線形の",
    "logarithmic": "対数の",
    "longitudinal": "縦方向の",
    "macroscopic": "巨視的な",
    "marginal": "限界の",
    "mechanical": "機械的な",
    "microscopic": "微視的な",
    "minimal": "最小の",
    "modular": "モジュール式の",
    "mutual": "相互の",
    "numerical": "数値の",
    "objective": "客観的な",
    "optimal": "最適な",
    "parallel": "平行の",
    "periodic": "周期的な",
    "peripheral": "周辺の",
    "plastic": "塑性の",
    "predictive": "予測的な",
    "preliminary": "予備的な",
    "primary": "主要な",
    "principal": "主要な",
    "progressive": "漸進的な",
    "prospective": "将来の",
    "radial": "半径方向の",
    "random": "無作為の",
    "reciprocal": "逆数の",
    "redundant": "冗長な",
    "relative": "相対的な",
    "reliable": "信頼性のある",
    "residual": "残留の",
    "resilient": "弾力性のある",
    "rotational": "回転の",
    "scalable": "拡張可能な",
    "selective": "選択的な",
    "sequential": "逐次の",
    "singular": "特異な",
    "spatial": "空間の",
    "subsequent": "後続の",
    "symmetric": "対称の",
    "tangential": "接線方向の",
    "temporal": "時間の",
    "uniform": "均一の",
    "variable": "可変の",
    "vectorial": "ベクトルの",
    "volatile": "揮発性の",
    # ── Adverbs ─────────────────────────────
    "efficiently": "効率的に",
}

# Stem-based meanings are ONLY used for compound terms (with space/hyphen)
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
    "biome": "生物群系",
}

# ━━━━ カタカナのみ翻訳を漢字/ひらがなに修正 ━━━━
# alkana が生成した不自然なカタカナ変換を、正しい日本語に置換する
BASE_JA_MEANINGS.update({
    # ── 基本科学・工学用語 ──
    "acoustic": "音響の",
    "acoustical": "音響の",
    "acoustically": "音響的に",
    "acoustician": "音響学者",
    "actuator": "作動装置",
    "acupressure": "指圧",
    "admittance": "アドミタンス（容量）",
    "aero": "航空の",
    "aerobatics": "曲技飛行",
    "aerobic": "好気性の",
    "aerobics": "有酸素運動",
    "aeronautic": "航空の",
    "aeronautical": "航空の",
    "aeronautics": "航空学",
    "aeroplane": "飛行機",
    "aeroplanes": "飛行機",
    "aerosol": "エアロゾル（噴霧）",
    "aerosols": "エアロゾル（噴霧）",
    "aerosolized": "エアロゾル化された",
    "aerospace": "航空宇宙",
    "allergenic": "アレルゲン性の",
    "altimeter": "高度計",
    "amphibology": "両義性",
    "anaerobic": "嫌気性の",
    "analogical": "類推の",
    "anemometer": "風速計",
    "anemometry": "風速測定",
    "anesthesiology": "麻酔学",
    "angiography": "血管造影",
    "anthology": "選集",
    "anthropogenic": "人為的な",
    "anthropology": "人類学",
    "anthropometry": "人体計測",
    "antigenic": "抗原性の",
    "archaeology": "考古学",
    "archeology": "考古学",
    "astrology": "占星術",
    "asymmetry": "非対称性",
    "autobiography": "自伝",
    "bandgap": "バンドギャップ（禁制帯幅）",
    "benchmarking": "ベンチマーク評価",
    "betatron": "ベータトロン（電子加速器）",
    "bibliography": "参考文献",
    "bioclimatology": "生物気候学",
    "bioenergy": "生物エネルギー",
    "biogenic": "生物由来の",
    "biography": "伝記",
    "biology": "生物学",
    "biome": "生物群系",
    "biomes": "生物群系",
    "biomed": "生物医学",
    "biomedicine": "生物医学",
    "biometric": "生体認証の",
    "biometry": "生物統計学",
    "biosystematic": "生物分類学の",
    "biosystems": "生物システム",
    "biotechnology": "生物工学",
    "bivalve": "二枚貝",
    "bivalves": "二枚貝",
    "butt-weld": "突合せ溶接",
    "calligraphy": "書道",
    "calorimeter": "熱量計",
    "carcinogenic": "発癌性の",
    "carload": "車一台分の荷",
    "carloads": "車一台分の荷",
    "centimeter": "センチメートル（長さの単位）",
    "ceramic": "セラミック（窯業製品）",
    "ceramics": "セラミックス（窯業）",
    "checksum": "検査合計",
    "choreography": "振付",
    "chronology": "年代学",
    "chronometer": "クロノメーター（精密計時器）",
    "cinematography": "撮影技術",
    "climatology": "気候学",
    "coefficient": "係数",
    "coefficients": "係数",
    "columnar": "円柱状の",
    "columnist": "コラムニスト（論説委員）",
    "columnists": "コラムニスト",
    "compliance": "準拠性",
    "computer": "コンピュータ（計算機）",
    "computerization": "コンピュータ化",
    "computerized": "コンピュータ化された",
    "computers": "コンピュータ（計算機）",
    "concrete": "コンクリート（建設材料）",
    "concurrently": "同時に",
    "conductance": "コンダクタンス（電気伝導率）",
    "constructive": "建設的な",
    "constructor": "構築者",
    "constructors": "構築者",
    "controller": "制御装置",
    "controllers": "制御装置",
    "coprocessor": "コプロセッサ（補助演算装置）",
    "cosmetology": "美容学",
    "cosmology": "宇宙論",
    "cosmotron": "コスモトロン（陽子加速器）",
    "cost-efficient": "費用効率の良い",
    "creep": "クリープ（塑性変形）",
    "criminology": "犯罪学",
    "cybertron": "サイバトロン（架空）",
    "cyclotron": "サイクロトロン（粒子加速器）",
    "data": "データ（情報）",
    "data-based": "データに基づく",
    "databank": "データバンク（情報基盤）",
    "database": "データベース（情報管理系）",
    "databases": "データベース",
    "datable": "年代測定可能な",
    "datacenter": "データセンター（情報処理施設）",
    "datacenters": "データセンター",
    "datas": "データ",
    "dataset": "データセット（情報集合）",
    "datasets": "データセット",
    "datasheet": "データシート（仕様書）",
    "deconstruct": "脱構築する",
    "deconstructed": "脱構築された",
    "deconstructing": "脱構築中の",
    "deconstruction": "脱構築",
    "deconstructs": "脱構築する",
    "densitometer": "濃度計",
    "designee": "被指名者",
    "destructible": "破壊可能な",
    "destructive": "破壊的な",
    "dialysis": "透析",
    "diameter": "直径",
    "digital": "デジタル（電子的な）",
    "digitalis": "ジギタリス（強心配糖体）",
    "digitally": "デジタル的に",
    "dissimulation": "偽装",
    "downloadable": "ダウンロード可能な",
    "doxology": "頌栄",
    "dynamical": "力学の",
    "dynamically": "動的に",
    "ecology": "生態学",
    "ecstatic": "恍惚とした",
    "efficient": "効率的な",
    "electrolysis": "電気分解",
    "electron": "電子",
    "embryology": "発生学",
    "endocrinology": "内分泌学",
    "endoscope": "内視鏡",
    "energy": "エネルギー（活力）",
    "energy-absorbing": "エネルギー吸収の",
    "energy-releasing": "エネルギー放出の",
    "energy-storing": "エネルギー貯蔵の",
    "energys": "エネルギー",
    "entomology": "昆虫学",
    "entropy": "エントロピー（乱雑度）",
    "epidemiology": "疫学",
    "epistemology": "認識論",
    "ergometer": "エルゴメーター（仕事量計）",
    "eschatology": "終末論",
    "ethnology": "民族学",
    "ethology": "動物行動学",
    "etiology": "病因学",
    "etymology": "語源学",
    "eugenic": "優生学の",
    "fabric": "布地",
    "fabrication": "製作",
    "fabrications": "製作",
    "fabricator": "製作者",
    "fabricators": "製作者",
    "fabrics": "布地",
    "fathometer": "水深計",
    "feedback": "フィードバック（帰還）",
    "feedforward": "フィードフォワード（前送り）",
    "ferroconcrete": "鉄筋コンクリート",
    "filesystem": "ファイルシステム（情報管理体系）",
    "firmware": "ファームウェア（組込みソフト）",
    "flowchart": "フローチャート（流れ図）",
    "flown": "飛んだ",
    "fluid": "流体",
    "fluidity": "流動性",
    "fluids": "流体",
    "frame": "枠組み",
    "frame-up": "でっち上げ",
    "futurology": "未来学",
    "genic": "遺伝子の",
    "geography": "地理学",
    "geology": "地質学",
    "geometry": "幾何学",
    "gerontology": "老年学",
    "gynecology": "婦人科学",
    "hemodynamic": "血行力学の",
    "heuristics": "発見的手法",
    "hexameter": "六歩格",
    "high-energy": "高エネルギーの",
    "high-flown": "大げさな",
    "histology": "組織学",
    "horoscope": "星占い",
    "hydrology": "水文学",
    "hydrometer": "比重計",
    "iatrogenic": "医原性の",
    "ichthyology": "魚類学",
    "iconography": "図像学",
    "ideology": "イデオロギー（思想体系）",
    "immaterial": "非物質的な",
    "immunogenic": "免疫原性の",
    "impedance": "インピーダンス（交流抵抗）",
    "inductance": "インダクタンス（誘導係数）",
    "inefficiencies": "非効率",
    "inefficiency": "非効率",
    "inefficient": "非効率な",
    "inelastic": "非弾性の",
    "infrastructural": "社会基盤の",
    "instructional": "教育の",
    "instructive": "教育的な",
    "instructress": "女性教官",
    "interferometry": "干渉測定法",
    "jumbotron": "大型表示装置",
    "kaleidoscope": "万華鏡",
    "kilogram-meter": "キログラムメートル（仕事の単位）",
    "kilometer": "キロメートル（長さの単位）",
    "kinesiology": "運動学",
    "laparoscope": "腹腔鏡",
    "lexicography": "辞書学",
    "limnology": "陸水学",
    "loader": "装填装置",
    "loaders": "装填装置",
    "machinations": "策略",
    "machinery": "機械類",
    "macrophage": "マクロファージ（大食細胞）",
    "mainframe": "メインフレーム（大型計算機）",
    "mainframes": "メインフレーム",
    "mammography": "マンモグラフィー（乳房撮影）",
    "manufacture": "製造",
    "manufactured": "製造された",
    "manufacturer": "製造業者",
    "manufacturers": "製造業者",
    "manufactures": "製造",
    "materialisation": "実体化",
    "materialise": "実体化する",
    "materialism": "唯物論",
    "materialistic": "物質主義的な",
    "materialists": "唯物論者",
    "materialization": "実体化",
    "matron": "婦長",
    "mechanic": "機械工",
    "mechanically": "機械的に",
    "mechanics": "力学",
    "mechanistic": "機械論的な",
    "mechanization": "機械化",
    "mechatronics": "メカトロニクス（機械電子工学）",
    "metadata": "メタデータ（付随情報）",
    "metamaterials": "メタマテリアル（人工物質）",
    "metastatic": "転移性の",
    "meter": "計器",
    "methodology": "方法論",
    "micro": "微小の",
    "micro-organism": "微生物",
    "microalgae": "微小藻類",
    "microarray": "マイクロアレイ（微小配列）",
    "microarrays": "マイクロアレイ",
    "microbalance": "微量天秤",
    "microbar": "マイクロバール（圧力単位）",
    "microbat": "小型コウモリ",
    "microbe": "微生物",
    "microbeads": "微小ビーズ",
    "microbial": "微生物の",
    "microbic": "微生物の",
    "microbiological": "微生物学の",
    "microbiologist": "微生物学者",
    "microbiologists": "微生物学者",
    "microbiome": "微生物叢",
    "microbreweries": "小規模醸造所",
    "microbrewery": "小規模醸造所",
    "microcephalic": "小頭症の",
    "microchip": "マイクロチップ（集積回路）",
    "microchipped": "マイクロチップ装着済み",
    "microchips": "マイクロチップ",
    "microclimate": "微気候",
    "microcode": "マイクロコード（機械語）",
    "microcomputer": "マイクロコンピュータ（小型計算機）",
    "microcomputers": "マイクロコンピュータ",
    "microcontroller": "マイクロコントローラ（制御用IC）",
    "microcontrollers": "マイクロコントローラ",
    "microcosmic": "小宇宙の",
    "microcredit": "マイクロクレジット（少額融資）",
    "microcrystalline": "微結晶の",
    "microdot": "マイクロドット（微小印刷点）",
    "microeconomic": "ミクロ経済の",
    "microeconomist": "ミクロ経済学者",
    "microelectronic": "マイクロエレクトロニクスの",
    "microenvironment": "微小環境",
    "microevolution": "小進化",
    "microfiber": "極細繊維",
    "microfilm": "マイクロフィルム（縮小写真記録）",
    "microfinance": "マイクロファイナンス（少額金融）",
    "microflora": "微生物叢",
    "microfossil": "微化石",
    "microglia": "ミクログリア（小膠細胞）",
    "micrograms": "マイクログラム（質量単位）",
    "microgravity": "微小重力",
    "micromanage": "細部管理する",
    "micromanaging": "細部管理する",
    "micromax": "マイクロマックス",
    "micrometeor": "微小流星",
    "micrometeoric": "微小隕石の",
    "micrometeorite": "微小隕石",
    "micrometer": "マイクロメータ（精密測定器）",
    "micrometers": "マイクロメータ",
    "micrometry": "微小測定",
    "micromicron": "マイクロミクロン（長さ単位）",
    "micron": "ミクロン（長さ単位）",
    "micronor": "マイクロノア",
    "microns": "ミクロン",
    "micronutrient": "微量栄養素",
    "micronutrients": "微量栄養素",
    "micropenis": "矮小陰茎",
    "microphallus": "矮小陰茎",
    "microphone": "マイクロフォン（集音器）",
    "microphones": "マイクロフォン",
    "microphoning": "ハウリング",
    "microprocessor": "マイクロプロセッサ（中央処理装置）",
    "microprocessors": "マイクロプロセッサ",
    "microradian": "マイクロラジアン（角度単位）",
    "micros": "微小の",
    "microsatellite": "超小型衛星",
    "microscale": "微小規模",
    "microscope": "顕微鏡",
    "microscopes": "顕微鏡",
    "microscopical": "顕微鏡的な",
    "microsecond": "マイクロ秒（時間単位）",
    "microseconds": "マイクロ秒",
    "microservices": "マイクロサービス（分散処理構造）",
    "microsofts": "マイクロソフト",
    "microsome": "ミクロソーム（小胞体）",
    "microspheres": "微小球",
    "microspore": "小胞子",
    "microsurgery": "微小外科",
    "microsystems": "マイクロシステム（微小機械系）",
    "microtome": "ミクロトーム（薄切機）",
    "microvascular": "微小血管の",
    "microvolt": "マイクロボルト（電圧単位）",
    "middleware": "ミドルウェア（中間ソフト）",
    "millimeter": "ミリメートル（長さの単位）",
    "model": "模型",
    "moment": "モーメント（力の能率）",
    "monogenic": "単一遺伝子の",
    "morphology": "形態学",
    "multimeter": "マルチメーター（万能計器）",
    "musicology": "音楽学",
    "myology": "筋学",
    "mythology": "神話学",
    "nano": "ナノ（極微小）",
    "nanobots": "ナノロボット",
    "nanocephalic": "ナノ頭症の",
    "nanogram": "ナノグラム（質量単位）",
    "nanometer": "ナノメートル（長さ単位）",
    "nanometre": "ナノメートル",
    "nanos": "ナノ",
    "nanoscale": "ナノスケール（極微小規模）",
    "nanoscience": "ナノ科学",
    "nanosecond": "ナノ秒（時間単位）",
    "nanoseconds": "ナノ秒",
    "nanovolt": "ナノボルト（電圧単位）",
    "nanowire": "ナノワイヤ（極細導線）",
    "nanowires": "ナノワイヤ",
    "nanotech": "ナノテク（超微細技術）",
    "nanotube": "ナノチューブ（極細管）",
    "nanotubes": "ナノチューブ",
    "negatron": "陰電子",
    "network": "ネットワーク（通信網）",
    "neutron": "中性子",
    "obstructionist": "妨害者",
    "obstructive": "閉塞性の",
    "octameter": "八歩格",
    "odometer": "走行距離計",
    "odontology": "歯科学",
    "oenology": "醸造学",
    "oncogenic": "発癌性の",
    "ontology": "存在論",
    "optimisation": "最適化",
    "optimise": "最適化する",
    "optimising": "最適化中の",
    "optimism": "楽観主義",
    "optimist": "楽観主義者",
    "optimistically": "楽観的に",
    "optimists": "楽観主義者",
    "optimizer": "最適化器",
    "optometry": "検眼学",
    "ornithology": "鳥類学",
    "osteology": "骨学",
    "otolaryngology": "耳鼻咽喉科学",
    "otoscope": "耳鏡",
    "overflow": "溢流",
    "paleogeology": "古地質学",
    "paralysis": "麻痺",
    "parameter": "パラメータ（媒介変数）",
    "parasitology": "寄生虫学",
    "patron": "後援者",
    "payload": "有効荷重",
    "payloads": "有効荷重",
    "pedometer": "歩数計",
    "penology": "刑罰学",
    "pentameter": "五歩格",
    "perimeter": "周長",
    "periscope": "潜望鏡",
    "phenomenology": "現象学",
    "philology": "文献学",
    "photodiode": "フォトダイオード（光検出素子）",
    "photolithography": "フォトリソグラフィー（光転写製版）",
    "phraseology": "語法",
    "polygenic": "多因子性の",
    "polymerase": "ポリメラーゼ（重合酵素）",
    "positron": "陽電子",
    "posology": "用量学",
    "powerful": "強力な",
    "powerfully": "強力に",
    "preprocessor": "プリプロセッサ（前処理装置）",
    "procession": "行列",
    "processions": "行列",
    "proctology": "肛門病学",
    "prostatic": "前立腺の",
    "protocol": "プロトコル（通信規約）",
    "psephology": "選挙学",
    "quantum": "量子",
    "reactance": "リアクタンス（誘導抵抗）",
    "recurrent": "反復する",
    "remodel": "改造する",
    "remodeled": "改造された",
    "remodeling": "改造",
    "remodelled": "改造された",
    "remodelling": "改造",
    "rheology": "レオロジー（流動学）",
    "rheumatology": "リウマチ学",
    "robot": "ロボット（自動機械）",
    "robotic": "ロボット工学の",
    "robots": "ロボット",
    "sampling": "標本抽出",
    "scope": "範囲",
    "seamstress": "裁縫師",
    "seamstresses": "裁縫師",
    "semi-processed": "半加工の",
    "semiconducting": "半導体の",
    "semiconscious": "半意識の",
    "semifluidity": "半流動性",
    "sensor": "センサ（検出器）",
    "sensorial": "感覚の",
    "sensorimotor": "感覚運動の",
    "sensorium": "感覚中枢",
    "sensors": "センサ",
    "sensory": "感覚の",
    "serology": "血清学",
    "servo": "サーボ（自動制御機構）",
    "simulation": "シミュレーション（模擬実験）",
    "simulations": "シミュレーション",
    "sinology": "中国学",
    "spectroscope": "分光器",
    "spectrum": "スペクトル（分光分布）",
    "speedometer": "速度計",
    "sphygmomanometer": "血圧計",
    "spirometer": "肺活量計",
    "spot-weld": "スポット溶接",
    "spot-welder": "スポット溶接機",
    "steele": "鉄鋼の",
    "steely": "鋼鉄のような",
    "steganography": "ステガノグラフィー（情報秘匿）",
    "stenography": "速記",
    "stereoscope": "立体鏡",
    "stethoscope": "聴診器",
    "strainer": "濾過器",
    "stressor": "ストレス要因",
    "stressors": "ストレス要因",
    "struct": "構造体",
    "structuralism": "構造主義",
    "structurally": "構造的に",
    "subframe": "サブフレーム（副枠）",
    "supercomputer": "スーパーコンピュータ（超大型計算機）",
    "supercomputers": "スーパーコンピュータ",
    "superfluid": "超流動体",
    "supermodels": "スーパーモデル",
    "superpowered": "超強力な",
    "symmetry": "対称性",
    "synchrotron": "シンクロトロン（粒子加速器）",
    "system": "システム（体系）",
    "systematic": "体系的な",
    "systematically": "体系的に",
    "systemic": "全身性の",
    "systemically": "全身的に",
    "systems": "システム",
    "taximeter": "タクシーメーター（料金計）",
    "teleology": "目的論",
    "telescope": "望遠鏡",
    "terminology": "用語",
    "theology": "神学",
    "throughput": "スループット（処理能力）",
    "timber-framed": "木造の",
    "tonometer": "眼圧計",
    "transistor": "トランジスタ（半導体素子）",
    "transistors": "トランジスタ",
    "trigonometry": "三角法",
    "tron": "トロン（電子管）",
    "turbine": "タービン（回転原動機）",
    "turbines": "タービン",
    "urinalysis": "尿検査",
    "value-system": "価値体系",
    "valve": "弁",
    "valves": "弁",
    "virology": "ウイルス学",
    "vulcanology": "火山学",
    "weld": "溶接",
    "welded": "溶接された",
    "welder": "溶接工",
    "welders": "溶接工",
    "weldon": "溶接の",
    "welds": "溶接",
    "zoology": "動物学",
    # もともとkatakanaでOKだが説明付きを追加
    "beaman": "梁工",
    "beamer": "梁材",
    "bedframe": "寝台枠",
    "mistresses": "女性支配者",
    "metatron": "メタトロン（架空）",
    "textron": "テクストロン（会社名）",
    "voltron": "ボルトロン（架空）",
    "doxology": "頌栄",
    "stratigraphy": "層序学",
    "photonics": "フォトニクス（光工学）",
    # ── Remaining vague fixes (batch 3) ──
    # Adjectives
    "acrogenic": "頂端成長の",
    "aerobiotic": "好気性の",
    "androgenic": "男性ホルモンの",
    "angiogenic": "血管新生の",
    "constructional": "構造上の",
    "cosmogenic": "宇宙線生成の",
    "cyanogenic": "シアン生成の",
    "endogenic": "内因性の",
    "fabricant": "製造者",
    "fluidic": "流体の",
    "hematogenic": "血液生成の",
    "metamaterial": "メタマテリアル（人工物質）",
    "microwavable": "電子レンジ対応の",
    "microwaveable": "電子レンジ対応の",
    "myogenic": "筋原性の",
    "osteogenic": "骨形成の",
    "psychogenic": "心因性の",
    "pyogenic": "化膿性の",
    "somatogenic": "体因性の",
    # Adverb
    "abeam": "正横に",
    # Science -ology words
    "agroecology": "農業生態学",
    "audiology": "聴覚学",
    "cryptology": "暗号学",
    "demography": "人口統計学",
    "egyptology": "エジプト学",
    "glaciology": "氷河学",
    "graphology": "筆跡学",
    "herbology": "薬草学",
    "iconology": "図像学",
    "nosology": "疾病分類学",
    "oceanology": "海洋学",
    "paleology": "古生物学",
    "phenology": "生物季節学",
    "phrenology": "骨相学",
    "sedimentology": "堆積学",
    "symptomatology": "症候学",
    # -graphy words
    "arteriography": "動脈造影",
    "arthrography": "関節造影",
    "autoradiography": "オートラジオグラフィー（自己放射線撮影）",
    "cardiography": "心電図記録",
    "echography": "超音波検査",
    "encephalography": "脳波検査",
    "epigraphy": "碑文学",
    "ethnography": "民族誌学",
    "lymphography": "リンパ管造影",
    "orthography": "正字法",
    "paleography": "古文書学",
    "photogrammetry": "写真測量",
    "physiography": "自然地理学",
    "planography": "平版印刷",
    "polarography": "ポーラログラフィー（電気化学分析）",
    "pyelography": "腎盂造影",
    "radiography": "放射線撮影",
    "tachygraphy": "速記",
    "videography": "映像制作",
    # -scope words
    "angioscope": "血管内視鏡",
    "arthroscope": "関節鏡",
    "bronchoscope": "気管支鏡",
    "gastroscope": "胃カメラ",
    "keratoscope": "角膜鏡",
    "laryngoscope": "喉頭鏡",
    "orthoscope": "正視鏡",
    "rhinoscope": "鼻鏡",
    # -meter words
    "actinometer": "放射計",
    "audiometer": "聴力計",
    "craniometer": "頭蓋計",
    "decameter": "デカメートル（長さ単位）",
    "decimeter": "デシメートル（長さ単位）",
    "dosemeter": "線量計",
    "fluxmeter": "磁束計",
    "machmeter": "マッハ計",
    "microphotometer": "微小光度計",
    "ohmmeter": "抵抗計",
    "radiometer": "放射計",
    "sclerometer": "硬度計",
    "seismometer": "地震計",
    "spectrophotometer": "分光光度計",
    "tachymeter": "タキメーター（距離測定器）",
    "tetrameter": "四歩格",
    "wattmeter": "電力計",
    "eurobarometer": "世論調査計",
    # -metry words
    "alkalimetry": "アルカリ測定",
    "astrometry": "位置天文学",
    "craniometry": "頭蓋測定",
    "oximetry": "酸素飽和度測定",
    "psychometry": "精神測定",
    "sociometry": "社会計量学",
    "viscometry": "粘度測定",
    # -lysis words
    "hematolysis": "溶血",
    "lipolysis": "脂肪分解",
    "osteolysis": "骨溶解",
    "radiolysis": "放射線分解",
    "rhabdomyolysis": "横紋筋融解症",
    "thrombolysis": "血栓溶解",
    "epidermolysis": "表皮水疱症",
    "cycloaddition": "環化付加",
    # Aero- words
    "aeroembolism": "空気塞栓症",
    "aerofoil": "翼型",
    "aerogel": "エアロゲル（極低密度物質）",
    "aerogenerator": "風力発電機",
    "aerogram": "航空書簡",
    "aerolite": "石質隕石",
    "aeromedicine": "航空医学",
    "aerophagia": "空気嚥下症",
    "aerobically": "有酸素的に",
    # Data- words
    "dataflow": "データフロー（情報流れ）",
    "datagram": "データグラム（通信単位）",
    "datalink": "データリンク（通信接続）",
    "datamining": "データマイニング（情報抽出）",
    "datapoint": "データポイント（測定点）",
    "datatype": "データ型",
    # Micro- words
    "microgram": "マイクログラム（質量単位）",
    "micrograph": "顕微鏡写真",
    "micrographs": "顕微鏡写真",
    # Nano- words
    "nanomedicine": "ナノ医療",
    # Other technical
    "algorithmically": "アルゴリズム的に",
    "pharmacodynamics": "薬力学",
    "machination": "策略",
    "robotically": "ロボット的に",
    "reflow": "リフロー（再溶融）",
    "digitalized": "デジタル化された",
    "overstressed": "過負荷の",
    "reprocess": "再処理する",
    "systematize": "体系化する",
    "steelwork": "鉄骨工事",
    # ── Katakana-only fixes (add kanji explanation) ──
    "algorithm": "アルゴリズム（算法）",
    "algorithms": "アルゴリズム（算法）",
    "bolometer": "ボロメーター（放射線検出器）",
    "cashflow": "キャッシュフロー（資金繰り）",
    "chromatography": "クロマトグラフィー（分離分析法）",
    "cohomology": "コホモロジー（数学）",
    "destructor": "デストラクタ（解放処理）",
    "etching": "エッチング（腐食加工）",
    "gasometer": "ガスタンク（気体貯蔵槽）",
    "gaussmeter": "ガウスメーター（磁場測定器）",
    "gyroscope": "ジャイロスコープ（回転儀）",
    "holography": "ホログラフィー（立体写真術）",
    "iconoscope": "アイコノスコープ（撮像管）",
    "intron": "イントロン（非翻訳配列）",
    "katharometer": "カタロメーター（熱伝導検出器）",
    "klystron": "クライストロン（マイクロ波管）",
    "lithography": "リソグラフィー（石版印刷）",
    "magnetron": "マグネトロン（マイクロ波発生器）",
    "oscilloscope": "オシロスコープ（波形観測器）",
    "permalloy": "パーマロイ（磁性合金）",
    "permeance": "パーミアンス（透磁率）",
    "phage": "ファージ（ウイルス）",
    "planimeter": "プラニメーター（面積計）",
    "pump": "ポンプ（送液装置）",
    "quercitron": "ケルシトロン（黄色染料）",
    "rheometer": "レオメーター（粘弾性測定器）",
    "stroboscope": "ストロボスコープ（回転速度計）",
    "synchrocyclotron": "シンクロサイクロトロン（粒子加速器）",
    "tacheometer": "タキメーター（距離角測定器）",
    "tachistoscope": "タキストスコープ（瞬間露出器）",
    "telemetry": "テレメトリー（遠隔測定）",
    "tribology": "トライボロジー（摩擦学）",
    "truss": "トラス（骨組構造）",
    "typographer": "タイポグラファー（活版職人）",
    "typography": "タイポグラフィー（活版印刷術）",
    "variometer": "バリオメーター（磁気変動計）",
    "wireframe": "ワイヤーフレーム（骨格模型）",
    "workflow": "ワークフロー（作業手順）",
    "workflows": "ワークフロー（作業手順）",
    "xerography": "ゼログラフィー（乾式複写）",
    "xeroradiography": "ゼロラジオグラフィー（乾式X線撮影）",
    "bevatron": "ベバトロン（粒子加速器）",
    "cistron": "シストロン（遺伝子単位）",
    "chylomicron": "カイロミクロン（脂質粒子）",
    # ── Batch 4 vague fixes ──
    "antistatic": "帯電防止の",
    "cardiogenic": "心臓性の",
    "chromogenic": "発色性の",
    "deconstructive": "脱構築的な",
    "hypostatic": "沈降性の",
    "microglial": "ミクログリアの",
    "overcurrent": "過電流の",
    "pharmacodynamic": "薬力学の",
    "radiogenic": "放射性の",
    "mechanistically": "機械論的に",
    "recurrently": "反復して",
    "anaerobically": "嫌気性的に",
    "datacentre": "データセンター（情報処理施設）",
    "deontology": "義務論",
    "framebuffer": "フレームバッファ（画像記憶装置）",
    "frameshift": "フレームシフト（読み枠変異）",
    "heterostructures": "ヘテロ構造",
    "nanopore": "ナノポア（極微小孔）",
    "periodontology": "歯周病学",
    "pulmonology": "呼吸器学",
    "remanufacturing": "再製造",
    "scintigraphy": "シンチグラフィー（放射線診断）",
    "tevatron": "テバトロン（粒子加速器）",
    "underflow": "アンダーフロー（下位桁溢れ）",
    "voltammetry": "ボルタンメトリー（電気化学分析）",
    "windpower": "風力発電",
    "microplate": "マイクロプレート（微量反応器）",
    "polymerize": "重合する",
    # ── Remaining katakana-only: add kanji explanation ──
    "analogue": "アナログ（類似物）",
    "analogues": "アナログ（類似物）",
    "bacteriophage": "バクテリオファージ（細菌ウイルス）",
    "bioreactor": "バイオリアクター（生物反応器）",
    "biosensor": "バイオセンサ（生物検出器）",
    "biosensors": "バイオセンサ（生物検出器）",
    "databases": "データベース（情報管理系）",
    "datacenters": "データセンター（情報処理施設）",
    "datasets": "データセット（情報集合）",
    "dekameter": "デカメートル（長さ単位）",
    "digitalin": "ジギタリン（強心配糖体）",
    "femtometer": "フェムトメートル（長さ単位）",
    "hectometer": "ヘクトメートル（長さ単位）",
    "mainframes": "メインフレーム（大型計算機）",
    "microarrays": "マイクロアレイ（微小配列）",
    "microchannel": "マイクロチャネル（微小流路）",
    "microchips": "マイクロチップ（集積回路）",
    "microcomputers": "マイクロコンピュータ（小型計算機）",
    "microcontrollers": "マイクロコントローラ（制御用IC）",
    "microelectronics": "マイクロエレクトロニクス（微小電子工学）",
    "microfarad": "マイクロファラッド（静電容量単位）",
    "microfiche": "マイクロフィッシュ（縮小写真）",
    "microgauss": "マイクロガウス（磁束密度単位）",
    "microgrid": "マイクログリッド（小規模電力網）",
    "microgrids": "マイクログリッド（小規模電力網）",
    "micromanagement": "マイクロマネジメント（細部管理）",
    "micrometers": "マイクロメータ（精密測定器）",
    "micromillimetre": "マイクロミリメートル（長さ単位）",
    "microns": "ミクロン（長さ単位）",
    "microphones": "マイクロフォン（集音器）",
    "microprocessors": "マイクロプロセッサ（中央処理装置）",
    "microtaggant": "マイクロタガント（微小追跡子）",
    "milliammeter": "ミリアンメーター（電流計）",
    "millimicron": "ミリミクロン（長さ単位）",
    "millivoltmeter": "ミリボルトメーター（電圧計）",
    "minicomputer": "ミニコンピュータ（小型計算機）",
    "minicomputers": "ミニコンピュータ（小型計算機）",
    "myriameter": "ミリアメートル（長さ単位）",
    "nanobots": "ナノロボット（極微小機械）",
    "nanoelectronics": "ナノエレクトロニクス（極微小電子工学）",
    "nanofibers": "ナノファイバー（極細繊維）",
    "nanometers": "ナノメートル（長さ単位）",
    "nanometre": "ナノメートル（長さ単位）",
    "nanotechnologies": "ナノテクノロジー（超微細技術）",
    "nanotechnology": "ナノテクノロジー（超微細技術）",
    "nanotubes": "ナノチューブ（極細管）",
    "nanowires": "ナノワイヤ（極細導線）",
    "picometer": "ピコメートル（長さ単位）",
    "robots": "ロボット（自動機械）",
    "sensors": "センサ（検出器）",
    "servosystem": "サーボシステム（自動制御系）",
    "simulations": "シミュレーション（模擬実験）",
    "subsystem": "サブシステム（副系統）",
    "subsystems": "サブシステム（副系統）",
    "supercomputers": "スーパーコンピュータ（超大型計算機）",
    "systems": "システム（体系）",
    "transistors": "トランジスタ（半導体素子）",
    "turbines": "タービン（回転原動機）",
    # ── Batch 5 engineering vague fixes ──
    "microchipping": "マイクロチップ埋め込み",
    "microstrip": "マイクロストリップ（帯状線路）",
    "microcirculation": "微小循環",
    "microprobe": "微小探針",
    "microcapsules": "マイクロカプセル（微小被覆粒）",
    "microdosing": "微量投与",
    "phage": "ファージ（溶菌性の病原体）",
    # ── Batch 6 engineering vague fixes ──
    "chromodynamics": "色力学",
    "nanosheets": "ナノシート（極薄膜）",
    "flowsheet": "フローシート（工程図）",
    "micronized": "微粉化された",
    "micropayments": "マイクロペイメント（少額決済）",
    "postprocessing": "後処理",
    "velocimetry": "速度測定法",
    "datastore": "データストア（情報格納庫）",
    "microtonal": "微分音の",
    # ── Batch 7 engineering vague fixes ──
    "microburst": "微小突風",
    "micrometres": "マイクロメートル（長さ単位）",
    "microarchitecture": "マイクロアーキテクチャ（微細構成）",
    "unloader": "荷降ろし装置",
    "computerisation": "コンピュータ化（計算機化）",
    "microbus": "マイクロバス（小型バス）",
    # ── Batch 8 final vague fixes ──
    "databook": "データブック（資料集）",
    "overvoltage": "過電圧",
    "microsurgical": "顕微外科の",
    "odontogenic": "歯原性の",
    "signally": "著しく",
})

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
    "smith",
    "shit",
    "fuck",
    "porn",
    "porn",
    "trump",
    "bible",
    "church",
    "ball", "lifting", "baseball", "football",
}

# Morphological decomposition for technical compound words
TECH_PREFIX_JA = {
    "micro": "マイクロ",
    "nano": "ナノ",
    "macro": "マクロ",
    "super": "スーパー",
    "sub": "サブ",
    "ultra": "超",
    "hydro": "水力",
    "thermo": "熱",
    "electro": "電気",
    "aero": "航空",
    "bio": "バイオ",
    "geo": "地質",
    "photo": "光",
    "auto": "自動",
    "semi": "セミ",
    "multi": "マルチ",
    "poly": "ポリ",
    "mono": "モノ",
    "cryo": "極低温",
    "cyber": "サイバー",
    "tele": "遠隔",
    "neuro": "神経",
    "magneto": "磁気",
    "piezo": "圧電",
    "ferro": "強磁性",
    "opto": "光",
    "spectro": "分光",
    "chrono": "時間",
    "para": "準",
    "dia": "反",
    "iso": "等",
    "inter": "相互",
    "trans": "変換",
    "proto": "原型",
    "pseudo": "擬似",
    "quasi": "準",
    "turbo": "ターボ",
    "pyro": "高温",
    "sono": "音波",
    "acousto": "音響",
}

TECH_SUFFIX_JA = {
    "ology": "学",
    "logy": "学",
    "ography": "記録法",
    "graphy": "法",
    "metry": "測定法",
    "metric": "測定の",
    "scopy": "鏡検査",
    "scope": "鏡",
    "meter": "計",
    "ics": "工学",
    "ism": "主義",
    "ist": "工",
    "ical": "的な",
    "lysis": "分解",
    "genic": "発生の",
    "tron": "トロン",
    "ware": "ウェア",
    "genesis": "生成",
    "logical": "学的な",
    "logist": "学者",
}

# Known root parts for compound decomposition
ROOT_JA = {
    "chip": "チップ",
    "system": "システム",
    "computer": "コンピュータ",
    "structure": "構造",
    "conductor": "導体",
    "cosm": "宇宙",
    "scope": "鏡",
    "wave": "波",
    "phone": "音",
    "film": "フィルム",
    "flow": "流れ",
    "power": "発電",
    "frame": "フレーム",
    "load": "負荷",
    "static": "静的",
    "dynamic": "力学",
    "dynamics": "力学",
    "mechanics": "力学",
    "electronics": "エレクトロニクス",
    "controller": "コントローラ",
    "chemical": "化学",
    "chemistry": "化学",
    "biology": "生物学",
    "physics": "物理学",
    "medical": "医学",
    "organism": "生物",
    "organisms": "生物",
    "particle": "粒子",
    "particles": "粒子",
    "material": "材料",
    "thermal": "熱",
    "acoustics": "音響学",
    "magnetic": "磁気の",
    "electric": "電気の",
    "mechanical": "機械的",
    "technical": "工学の",
    "logical": "論理の",
    "metric": "測定の",
    "processing": "処理",
    "processor": "プロセッサ",
    "network": "ネットワーク",
    "circuit": "回路",
    "sensor": "センサ",
    "data": "データ",
    "set": "セット",
    "base": "ベース",
    "model": "モデル",
    "grid": "グリッド",
    "bus": "バス",
    "tube": "チューブ",
    "fiber": "ファイバー",
    "array": "アレイ",
    "cell": "セル",
    "node": "ノード",
    "link": "リンク",
    "path": "パス",
    "channel": "チャネル",
    "layer": "層",
    "field": "場",
    "source": "源",
    "device": "デバイス",
    "module": "モジュール",
    "unit": "ユニット",
    "component": "コンポーネント",
    "element": "要素",
    "interface": "インターフェース",
    "wire": "ワイヤ",
    "cable": "ケーブル",
    "disk": "ディスク",
    "code": "コード",
    "ware": "ウェア",
}


def try_morphological_translation(word: str) -> str | None:
    """Try to translate via prefix/suffix decomposition."""
    # Try prefix match (longest prefix first)
    for prefix in sorted(TECH_PREFIX_JA, key=len, reverse=True):
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            remainder = word[len(prefix):]
            # Check if remainder is a known word
            if remainder in BASE_JA_MEANINGS:
                ja = BASE_JA_MEANINGS[remainder]
                # Strip katakana-only from base meaning (use prefix + base)
                return TECH_PREFIX_JA[prefix] + ja
            if remainder in TOKEN_JA_MEANINGS:
                return TECH_PREFIX_JA[prefix] + TOKEN_JA_MEANINGS[remainder]
            if remainder in ROOT_JA:
                return TECH_PREFIX_JA[prefix] + ROOT_JA[remainder]
            # Try singularized remainder
            sing_r = singularize_token(remainder)
            if sing_r in BASE_JA_MEANINGS:
                return TECH_PREFIX_JA[prefix] + BASE_JA_MEANINGS[sing_r]
            if sing_r in TOKEN_JA_MEANINGS:
                return TECH_PREFIX_JA[prefix] + TOKEN_JA_MEANINGS[sing_r]
            if sing_r in ROOT_JA:
                return TECH_PREFIX_JA[prefix] + ROOT_JA[sing_r]
            # Try suffix on remainder (e.g., bio + logy → バイオ + 学)
            for suffix in sorted(TECH_SUFFIX_JA, key=len, reverse=True):
                if remainder.endswith(suffix):
                    if remainder == suffix or len(remainder) == len(suffix):
                        # remainder IS the suffix (e.g., "logy" from bio+logy)
                        return TECH_PREFIX_JA[prefix] + TECH_SUFFIX_JA[suffix]
                    elif len(remainder) > len(suffix) + 1:
                        middle = remainder[:-len(suffix)]
                        # Try translating the middle root
                        mid_ja = ROOT_JA.get(middle) or BASE_JA_MEANINGS.get(middle) or TOKEN_JA_MEANINGS.get(middle)
                        if mid_ja:
                            return TECH_PREFIX_JA[prefix] + mid_ja + TECH_SUFFIX_JA[suffix]
                        # If middle is very short, just combine prefix+suffix
                        if len(middle) <= 3:
                            return TECH_PREFIX_JA[prefix] + TECH_SUFFIX_JA[suffix]
            # Do NOT fall through to alkana — return None so caller picks up
            # a better translation method instead of producing katakana garbage
    # Try suffix match (longest suffix first)
    for suffix in sorted(TECH_SUFFIX_JA, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            root = word[:-len(suffix)]
            # Try known dictionaries first
            root_ja = ROOT_JA.get(root) or BASE_JA_MEANINGS.get(root) or TOKEN_JA_MEANINGS.get(root)
            if root_ja:
                return root_ja + TECH_SUFFIX_JA[suffix]
            # Do NOT use alkana for root — produces bad katakana
    return None

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
    # Preserve underscore-separated compound words as spaces
    w = w.replace("_", " ")
    w = re.sub(r"\[[^\]]*\]", "", w)
    w = re.sub(r"\([^)]*\)", "", w)
    w = w.replace("–", "-").replace("—", "-")
    w = re.sub(r"[^a-z\- ]", "", w)
    w = re.sub(r"\s+", " ", w).strip()
    return w


def lemmatize_verb(word: str) -> str:
    """Return the base form of an English verb."""
    lemma = wn.morphy(word, "v")
    if lemma:
        return lemma
    # Manual suffix stripping
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if base + "e" in ADVANCED_EXAM_WORDS or base + "e" in BASE_JA_MEANINGS:
            return base + "e"
        if base in ADVANCED_EXAM_WORDS or base in BASE_JA_MEANINGS:
            return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if base in ADVANCED_EXAM_WORDS or base in BASE_JA_MEANINGS:
            return base
        if base + "e" in ADVANCED_EXAM_WORDS or base + "e" in BASE_JA_MEANINGS:
            return base + "e"
    if word.endswith("d") and len(word) > 3:
        base = word[:-1]
        if base in ADVANCED_EXAM_WORDS or base in BASE_JA_MEANINGS:
            return base
    return word


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
    """Normalize to canonical form for deduplication.
    Handles singular/plural AND verb inflections."""
    parts = re.split(r"[- ]+", word)
    if len(parts) == 1:
        w = parts[0]
        # Try singularize
        sing = singularize_token(w)
        # Try verb lemmatize
        verb_base = lemmatize_verb(w)
        # Pick the shortest (most base) form
        candidates = [w, sing, verb_base]
        return min(candidates, key=len)
    # Multi-word: singularize each part
    normed = [singularize_token(p) for p in parts if p and len(p) > 1]
    return " ".join(normed)


def choose_better_word(a: str, b: str) -> str:
    score_a = score_word(a) + (0.3 if a in ADVANCED_EXAM_WORDS else 0.0) - (0.1 if "-" in a else 0.0)
    score_b = score_word(b) + (0.3 if b in ADVANCED_EXAM_WORDS else 0.0) - (0.1 if "-" in b else 0.0)
    # Prefer base form (shorter) over inflected forms
    if len(a) < len(b) and score_a >= score_b - 0.5:
        return a
    if len(b) < len(a) and score_b >= score_a - 0.5:
        return b
    if score_b > score_a:
        return b
    return a


def is_usable(word: str) -> bool:
    if not word:
        return False
    parts = word.split()
    if len(parts) > 3:
        return False
    if len(word) < 3 or len(word) > 24:
        return False
    if word.startswith("list") or word.endswith("list"):
        return False
    if word.count("-") > 1:
        return False
    if not re.match(r"^[a-z][a-z\- ]*$", word):
        return False
    # Filter single-word concatenated compounds (>18 chars, no space/hyphen)
    if len(word) > 18 and " " not in word and "-" not in word:
        return False
    return True


def is_technical_like(word: str) -> bool:
    if word in ADVANCED_EXAM_WORDS:
        return True
    if word in BASE_JA_MEANINGS:
        return True
    if word in STOPWORDS:
        return False
    if zipf_frequency(word, "en") > 6.1 and word not in ADVANCED_EXAM_WORDS:
        return False
    if any(bad in word for bad in BAD_NON_TECH_SUBSTRINGS):
        return False
    # For multi-word terms, require at least one stem match
    parts = word.split()
    if len(parts) > 1:
        stem_matches = sum(1 for p in parts for stem in ENGINEERING_STEMS if stem in p)
        return stem_matches >= 1
    # Single words
    if any(stem in word for stem in ENGINEERING_STEMS):
        return True
    if any(word.endswith(sfx) for sfx in TECHNICAL_SUFFIXES):
        return True
    return False


def is_verb_inflection(word: str) -> bool:
    """Check if a word is an inflected verb form whose base already exists."""
    base = lemmatize_verb(word)
    return base != word and (base in ADVANCED_EXAM_WORDS or base in BASE_JA_MEANINGS)


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
    """Collect single-word technical terms from WordNet (no compound lemmas)."""
    terms = set()
    for lemma in wn.all_lemma_names(pos="n"):
        # Skip compound lemma names (contain underscore = multi-word)
        if "_" in lemma:
            continue
        lw = safe_word(lemma)
        if not is_usable(lw) or lw in STOPWORDS:
            continue
        if is_technical_like(lw):
            terms.add(lw)

    for lemma in wn.all_lemma_names(pos="a"):
        if "_" in lemma:
            continue
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
    """Infer Japanese meaning with priority: cache > curated dict > compound stem > token > alkana."""
    if word in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[word]

    # 1. Direct lookup in curated dictionary
    if word in BASE_JA_MEANINGS:
        TRANSLATION_CACHE[word] = BASE_JA_MEANINGS[word]
        return BASE_JA_MEANINGS[word]

    # 2. Try base/lemma form lookup (for words like "optimized" → "optimize")
    base = lemmatize_verb(word)
    if base != word and base in BASE_JA_MEANINGS:
        val = BASE_JA_MEANINGS[base]
        TRANSLATION_CACHE[word] = val
        return val
    sing = singularize_token(word)
    if sing != word and sing in BASE_JA_MEANINGS:
        val = BASE_JA_MEANINGS[sing]
        TRANSLATION_CACHE[word] = val
        return val

    # 3. Split word into parts (handles compounds AND single words)
    parts = re.split(r"[- ]+", word)
    mapped_parts = []
    for p in parts:
        base_p = singularize_token(p)
        if base_p in BASE_JA_MEANINGS:
            mapped_parts.append(BASE_JA_MEANINGS[base_p])
        elif base_p in TOKEN_JA_MEANINGS:
            mapped_parts.append(TOKEN_JA_MEANINGS[base_p])
        elif p in BASE_JA_MEANINGS:
            mapped_parts.append(BASE_JA_MEANINGS[p])
        elif p in TOKEN_JA_MEANINGS:
            mapped_parts.append(TOKEN_JA_MEANINGS[p])
        else:
            # For compound terms, try stem matching per-part
            if len(parts) > 1:
                found = False
                for stem, ja in STEM_MEANINGS.items():
                    if stem in p:
                        mapped_parts.append(ja)
                        found = True
                        break
                if not found:
                    # Try ROOT_JA for the part
                    if base_p in ROOT_JA:
                        mapped_parts.append(ROOT_JA[base_p])
                    elif p in ROOT_JA:
                        mapped_parts.append(ROOT_JA[p])
            # Do NOT fallback to alkana for single words — it produces garbage katakana
    if mapped_parts:
        val = "・".join(mapped_parts)
        TRANSLATION_CACHE[word] = val
        return val

    # 4. Try morphological prefix/suffix decomposition
    morph = try_morphological_translation(word)
    if morph:
        TRANSLATION_CACHE[word] = morph
        return morph

    # 5. Fallback based on POS
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
    return meaning not in {
        "工学用語", "技術専門語", "工学で用いる動作", "工学で用いる性質",
        "工学で用いる様態", "技術的に実行する", "技術的な性質", "技術的な方法",
    }


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
    if word in BASE_JA_MEANINGS:
        bonus += 1.5
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
    vague_terms = {"工学用語", "技術専門語", "技術的に実行する", "技術的な性質", "技術的な方法"}

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

    # High-value engineering seed set
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

    # Filter: usable, technical, not a stopword, not a verb inflection
    filtered = []
    for w in candidates:
        if not is_usable(w) or not is_technical_like(w) or w in STOPWORDS:
            continue
        if is_verb_inflection(w):
            continue
        filtered.append(w)

    filtered = sorted(set(filtered), key=lambda x: (-score_word(x), x))

    # Canonical dedup
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
            if is_verb_inflection(lw):
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
            if w not in selected_set and has_specific_meaning(w):
                selected.append(w)
                selected_set.add(w)

    # If still under target, allow words with any non-empty katakana meaning
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
                f"| {r['word']} | {r['ipa']} | {r['pos']} | {r['meaning_ja']} | {r['example_en']} |\n"
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
    print(f"POS distribution: {dict(df['pos'].value_counts())}")

    # Diagnostic counts
    vague_count = sum(1 for _, r in df.iterrows() if not is_specific_meaning(r["meaning_ja"]))
    print(f"Vague meanings: {vague_count}")

    # Check for wrong stem translations in single words
    stem_check = {"struct": "構造", "load": "荷重", "power": "電力", "beam": "梁",
                   "strain": "ひずみ", "stress": "応力", "design": "設計"}
    wrong_stem = 0
    for _, r in df.iterrows():
        w, m = r["word"], r["meaning_ja"]
        if " " in w or "-" in w:
            continue
        for stem, wrong_ja in stem_check.items():
            if stem in w and m == wrong_ja and w != stem:
                wrong_stem += 1
                break
    print(f"Wrong stem translations: {wrong_stem}")

    print(csv_path)
    print(json_path)
    print(md_path)
    print(added_path)
    print(quality_path)
    print(quality_summary_path)


if __name__ == "__main__":
    main()
