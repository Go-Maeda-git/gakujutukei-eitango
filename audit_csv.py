import csv
import re

with open('data/engineering_vocab_2000.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 1. Concatenated compound words (no space/hyphen, len>15)
concat = [r for r in rows if len(r['word'])>15 and ' ' not in r['word'] and '-' not in r['word']]
print(f'Concatenated compounds (>15 chars, no space/hyphen): {len(concat)}')
for r in concat[:10]:
    print(f"  {r['word']} -> {r['meaning_ja']}")

# 2. Verb inflections
base_verbs = {r['word'] for r in rows if r['pos']=='verb'}
inflected = []
for r in rows:
    if r['pos'] != 'verb': continue
    w = r['word']
    for suffix in ['ed', 'ing', 'd']:
        if not w.endswith(suffix): continue
        base = w[:-len(suffix)]
        if base in base_verbs or base+'e' in base_verbs:
            inflected.append(w)
            break
print(f'\nVerb inflections with base form: {len(inflected)}')
for w in inflected[:15]:
    print(f'  {w}')

# 3. Wrong stem translations for single words
stem_map = {'struct': '構造', 'load': '荷重', 'power': '電力', 'beam': '梁', 'strain': 'ひずみ', 'stress': '応力', 'design': '設計'}
wrong_stem = []
for r in rows:
    w = r['word']
    m = r['meaning_ja']
    if ' ' in w or '-' in w: continue
    for stem, wrong_ja in stem_map.items():
        if stem in w and m == wrong_ja and w != stem:
            wrong_stem.append((w, m, r['pos']))
            break
print(f'\nWrong stem-derived translations: {len(wrong_stem)}')
for w, m, p in wrong_stem[:20]:
    print(f'  {w} ({p}) -> {m}')

# 4. Katakana-only translations
def is_katakana_only(s):
    has_kana = False
    for c in s:
        if '\u30A0' <= c <= '\u30FF':
            has_kana = True
        elif '\u3040' <= c <= '\u309F' or '\u4E00' <= c <= '\u9FFF':
            return False
        elif c in '・ー':
            continue
    return has_kana

kata_only = [(r['word'], r['meaning_ja'], r['pos']) for r in rows if is_katakana_only(r['meaning_ja'])]
print(f'\nKatakana-only translations: {len(kata_only)}')
for w, m, p in kata_only[:20]:
    print(f'  {w} ({p}) -> {m}')

# 5. Total problems summary
problems = set()
for r in concat: problems.add(r['word'])
for w in inflected: problems.add(w)
for w, m, p in wrong_stem: problems.add(w)
for w, m, p in kata_only: problems.add(w)
print(f'\nTotal unique problematic words: {len(problems)}')
print(f'Total rows: {len(rows)}')
