# Engineering Vocab 2000 (GitHub Pages)

工学部編入試験向けの英語語彙学習アプリです。

## 内容
- 2000語の工学・科学系英単語データ
- 項目: 英単語 / IPA / 品詞 / 日本語の意味 / 例文
- 検索、品詞フィルタ、並び替え、ランダム5問クイズ

## ローカル確認
このフォルダをそのままブラウザで開くか、簡易サーバーを使って起動します。

```powershell
cd engineering-vocab-pages
../.venv/Scripts/python.exe -m http.server 8000
```

開くURL: `http://localhost:8000`

## データ再生成
```powershell
../.venv/Scripts/python.exe generate_vocab.py
```

## GitHub Pages 公開手順
1. GitHubで新規リポジトリ作成
2. このフォルダ内容をpush
3. GitHubの `Settings > Pages` で `Deploy from a branch` を選択
4. ブランチを `main`、フォルダを `/ (root)` に設定

公開URL例:
`https://<your-account>.github.io/<repo-name>/`
