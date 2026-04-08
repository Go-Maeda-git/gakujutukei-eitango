import csv, re

with open('data/engineering_vocab_2000.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

# Find katakana-only meanings (no kanji/hiragana)
kana_only = []
for r in rows:
    m = r['meaning_ja']
    has_kanji = bool(re.search(r'[一-龯]', m))
    has_hira = bool(re.search(r'[ぁ-ん]', m))
    has_kata = bool(re.search(r'[ァ-ヶー]', m))
    if has_kata and not has_kanji and not has_hira:
        kana_only.append(r)

print(f'Katakana-only translations: {len(kana_only)}')
for r in sorted(kana_only, key=lambda x: x['word']):
    print(f"  {r['word']:30s} {r['meaning_ja']}")
