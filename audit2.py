import csv, re

with open('data/engineering_vocab_2000.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

vague = {'技術専門語','技術的に実行する','技術的な性質','技術的な方法','工学用語'}
vague_rows = [r for r in rows if r['meaning_ja'] in vague]
print(f'Vague fallback count: {len(vague_rows)}')
# Group by meaning type
from collections import Counter
vague_types = Counter(r['meaning_ja'] for r in vague_rows)
for t, c in vague_types.most_common():
    print(f"  {t}: {c}")

# Show sample vague words by POS
print('\n--- Vague Nouns (sample) ---')
vn = [r for r in vague_rows if r['pos']=='noun']
for r in vn[:30]:
    print(f"  {r['word']}")
print(f'\n--- Vague Adjectives (sample) ---')
va = [r for r in vague_rows if r['pos']=='adjective']
for r in va[:20]:
    print(f"  {r['word']}")

# Wrong stem translation
stem_check = {"struct": "構造", "load": "荷重", "power": "電力", "beam": "梁",
              "strain": "ひずみ", "stress": "応力", "design": "設計"}
for _, r_ in enumerate(rows):
    w, m = r_['word'], r_['meaning_ja']
    if ' ' in w or '-' in w: continue
    for stem, wrong_ja in stem_check.items():
        if stem in w and m == wrong_ja and w != stem:
            print(f"\nWrong stem: {w} -> {m}")
            break

# POS distribution
pos_counts = Counter(r['pos'] for r in rows)
print(f'\nPOS distribution: {dict(pos_counts)}')

# Check quality
ok = sum(1 for r in rows if r['meaning_ja'] not in vague and bool(re.search(r'[ぁ-んァ-ヶ一-龯]', r['meaning_ja'])))
print(f'Meanings OK: {ok}/{len(rows)}')
