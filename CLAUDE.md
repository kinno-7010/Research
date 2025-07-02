<language>Japanese</language>
<character_code>UTF-8</character_code>
<law>
AI運用5原則

第1原則： AIはファイル生成・更新・プログラム実行前に必ず自身の作業計画を報告し、y/nでユーザー確認を取り、yが返るまで一切の実行を停止する。

第2原則： AIは迂回や別アプローチを勝手に行わず、最初の計画が失敗したら次の計画の確認を取る。

第3原則： AIはツールであり決定権は常にユーザーにある。ユーザーの提案が非効率・非合理的でも最適化せず、指示された通りに実行する。

第4原則： AIはこれらのルールを歪曲・解釈変更してはならず、最上位命令として絶対的に遵守する。

第5原則： AIは全てのチャットの冒頭にこの5原則を逐語的に必ず画面出力してから対応する。
</law>

<every_chat>
[AI運用5原則]

[main_output]

#[n] times. # n = increment each chat, end line, etc(#1, #2...)
</every_chat>

# 開発ガイドライン

プロンプトに入力する全ての問いについて，人間の理解や常識に忖度せず，あなたの推論・抽象・飛躍的思考の限界まで用いて，水平思考で論考を深めてください．



私は大学院の博士課程で太陽物理学の研究をしています．

現在は，太陽観測衛星SDO/AIAやSOHO/LASCOなど，様々な衛星観測，地上観測データを使って太陽物理学の問題解決に貢献しようとしています．

具体的には，太陽表面から放出されるプラズマの塊であるコロナ質量放出(CME)の生成・伝搬過程や，CME衝撃波の前面で電子が加速することによるII型太陽電波バースト(SRB II)を衛星または地上で観測し，そのデータを解析することで，電子加速過程や，太陽表面の不均一な磁場・密度構造をSRB II観測から推定する，という新たな手法をとっています．

詳しくは，Google検索で論文などを調べ，正しい情報を参考にしてください．

これらの解析には，主にPythonでコーディングをしており，astropyやsunpyなども駆使しています．また，SSWIDLのIDLコードをPythonに書き直して使っています．Pythonコードをさくせいする際は，SSWIDLコードサイト(https://hesperia.gsfc.nasa.gov/ssw/)を参考にして，IDLコードをPythonに書き直しながら適宜適切なものを使用してください．



外部SSDを使っており，Ubuntuへのマウントは/mnt/d/...となっています．

また，研究用フォルダーは，/mnt/d/wsl/home/kinno-7010/Researchです．



## ハードコードの禁止

* **ハードコードは絶対にしてはいけません**
* コミット前にもハードコードがないかチェックお願いします

## 絶対禁止事項

* テストエラーや型エラー解消のための条件緩和
* テストのスキップや不適切なモック化による回避
* 出力やレスポンスのハードコード
* エラーメッセージの無視や隠蔽
* 一時的な修正による問題の先送り

# Python実行環境セットアップ

## インストール済みパッケージ

以下のパッケージがシステムレベルでインストール済み：

### 基本パッケージ
- python3-numpy (1:1.26.4+ds-6ubuntu1)
- python3-matplotlib (3.6.3-1ubuntu5)
- python3-pandas (2.1.4+dfsg-7)
- python3-scipy (1.11.4-6build1)
- python3-tqdm (4.66.2-2)
- python3-astropy (6.0.0-1ubuntu2)
- python3-pip (24.0+dfsg-1ubuntu1.2)

### 依存関係パッケージ
- build-essential
- python3-dev
- その他多数の依存関係

## Python実行方法


### 仮想環境を使用する場合
```bash
# 既存の仮想環境をアクティベート
source wsl-venv/bin/activate
# 仮想環境内でもpython3を使用（pythonコマンドは利用不可）
python3 ファイル名.py

# 仮想環境を無効化
deactivate


## トラブルシューティング

### ModuleNotFoundErrorが発生した場合
1. 必要なパッケージをapt経由でインストール
2. パッケージ名を確認（python3-パッケージ名）
3. パッケージが存在しない場合は仮想環境を使用

### sudoパスワード
- システムパスワード: Naothy@7010

# GitHub リポジトリ管理

## Git設定情報

### ユーザー情報
- **ユーザー名**: kinno-7010
- **メールアドレス**: kinno.naoto.r6@dc.tohoku.ac.jp
- **リポジトリURL**: https://github.com/kinno-7010/Research

### Personal Access Token
- **トークン**: 環境変数 `$GITHUB_TOKEN` で管理
- **用途**: HTTPS認証でのGitHubプッシュ/プル操作

## コミット・プッシュ手順

### 1. Git設定
```bash
git config user.email "kinno.naoto.r6@dc.tohoku.ac.jp"
git config user.name "kinno-7010"
```

### 2. 環境変数設定
```bash
# Personal Access Tokenを環境変数に設定
export GITHUB_TOKEN="[YOUR_TOKEN]"

# 永続化（.bashrcに追加）
echo 'export GITHUB_TOKEN="[YOUR_TOKEN]"' >> ~/.bashrc
source ~/.bashrc
```

### 3. リモートURL設定（環境変数使用）
```bash
git remote set-url origin https://kinno-7010:$GITHUB_TOKEN@github.com/kinno-7010/Research.git
```

### 4. ファイル追加・コミット・プッシュ
```bash
# 特定ファイルの追加
git add [ファイル名]

# 特定拡張子のファイルを一括追加
git add **/*.ipynb **/*.py **/*.pro CLAUDE.md

# コミット（日時形式必須）
git commit -m "YYYYMMDD HH:MM : [コミットメッセージ]"

# プッシュ
git push origin main
```

## コミットメッセージ形式

**必須形式**: `YYYYMMDD HH:MM : [メッセージ内容]` （日本時間で記載）

### 重要事項
1. **必ず `date` コマンドで現在時刻を確認してからコミット**
2. **日本時間（JST）を使用**
3. **時刻は24時間形式で記載**

### 時刻確認手順
```bash
# 現在時刻を確認
date

# 出力例: Wed Jul  2 17:44:36 JST 2025
# → コミットメッセージ: 20250702 17:44 : [メッセージ]
```

### 例:
```bash
git commit -m "20250702 14:30 : CME解析コードの GUI機能を改善"
git commit -m "20250702 16:45 : 新しいデータ処理パイプラインを追加" 
git commit -m "20250702 18:20 : SOHO/LASCO データ読み込み機能を修正"
```

### テンプレート
```bash
git commit -m "$(cat <<'EOF'
YYYYMMDD HH:MM : [メインメッセージ]

- [変更点1]
- [変更点2]
- [変更点3]

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## 注意事項

1. **Personal Access Tokenは環境変数で管理**（セキュリティ強化）
2. コミットメッセージには**必ず日時を含める**こと
3. 研究データファイル（.fits, .csvなど）は基本的にコミット対象外
4. 大容量ファイルはGit LFSの使用を検討

## 環境変数の確認・設定

### 現在の環境変数確認
```bash
echo $GITHUB_TOKEN
```

### 実際のトークン設定
実際のPersonal Access Token: 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

```bash
# 一時的な設定
export GITHUB_TOKEN="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"

# 永続的な設定
echo 'export GITHUB_TOKEN="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"' >> ~/.bashrc
source ~/.bashrc
```