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
* ファイル・フォルダーの消去(消去を行う際は必ずユーザーに「何のファイル/フォルダーを何のために消去するのか」を示し，必ず確認を取ること．)

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
- **リポジトリURL**: git@github.com:kinno-7010/Research.git

## 🔐 認証設定

### SSH認証（推奨方法）

#### 1. SSH鍵設定確認
```bash
# SSH設定確認
cat ~/.ssh/config
# 出力例：
# Host github.com
#     HostName github.com
#     User git
#     IdentityFile ~/.ssh/id_ed25519_github
#     IdentitiesOnly yes

# SSH鍵ファイル確認
ls -la ~/.ssh/id_ed25519_github*
```

#### 2. SSH Agent設定
```bash
# SSH Agent起動・鍵追加
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github

# SSH接続テスト
ssh -T git@github.com
# 成功時出力: "Hi kinno-7010! You've successfully authenticated..."
```

#### 3. Git設定
```bash
# ユーザー情報設定
git config user.name "kinno-7010"
git config user.email "kinno.naoto.r6@dc.tohoku.ac.jp"

# リモートURL設定（SSH）
git remote set-url origin git@github.com:kinno-7010/Research.git
```

### Personal Access Token（代替方法）

#### 必要な新しいトークン作成手順
1. GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token (classic)" をクリック
3. **権限設定**:
   - ✅ `repo` (Full control of private repositories)
   - ✅ `workflow` (Update GitHub Action workflows)
   - ✅ `write:packages` (Upload packages to GitHub Package Registry)
4. **有効期限**: 90日（研究継続性を考慮）
5. トークンをコピーして保存

#### 環境変数設定
```bash
# 新しいトークンを環境変数に設定
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 永続化（.bashrcに追加）
echo 'export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc

# 設定確認
echo "Token設定確認: ${GITHUB_TOKEN:0:10}..."
```

#### Git設定（Token使用時）
```bash
# リモートURL設定（トークン認証）
git remote set-url origin https://kinno-7010:${GITHUB_TOKEN}@github.com/kinno-7010/Research.git

# 設定確認
git remote -v
```

## 📅 コミット・プッシュ手順


## 💡 推奨ワークフロー

### 日常的な作業パターン
```bash
# 1. 現在時刻確認
CURRENT_TIME=$(date '+%Y%m%d %H:%M')
echo "現在時刻: ${CURRENT_TIME}"

# 2. 変更ファイルの追加
git add [変更したファイル]

# 3. コミット
git commit -m "${CURRENT_TIME} : [具体的な変更内容]"

# 4. プッシュ
git push origin main
```

### シンプルなコミット・プッシュ
```bash
# ワンライナー
git add . && git commit -m "$(date '+%Y%m%d %H:%M') : [メッセージ]" && git push
```

## 🔧 Claude Code環境での特殊な問題と解決方法

### Claude Code環境でのSSH認証エラー

#### 典型的なエラー
```
Error: ssh_askpass: exec(/usr/bin/ssh-askpass): No such file or directory
git@github.com: Permission denied (publickey).
```

#### 原因
1. **対話的認証の制約**: Claude Code環境では、パスフレーズやパスワードの対話的入力ができません
2. **ssh-askpassの欠如**: WSL環境では `ssh-askpass` がデフォルトでインストールされていません
3. **SSH鍵のパスフレーズ**: 既存のSSH鍵にパスフレーズが設定されている場合、認証時に対話的入力が必要になります

#### 解決方法

##### Option A: 既存鍵からパスフレーズを削除
```bash
# 1. 既存鍵の確認
ls -la ~/.ssh/id_ed25519_github*

# 2. 既存鍵からパスフレーズを削除
ssh-keygen -p -f ~/.ssh/id_ed25519_github
# Enter old passphrase: [現在のパスフレーズを入力]
# Enter new passphrase (empty for no passphrase): [Enter]
# Enter same passphrase again: [Enter]

# 3. 公開鍵を確認（既にGitHubに登録済みの場合はスキップ）
cat ~/.ssh/id_ed25519_github.pub
```

##### Option B: 新規鍵を作成（古い鍵の場合）
```bash
# 1. 現在の鍵をバックアップ
mv ~/.ssh/id_ed25519_github ~/.ssh/id_ed25519_github.bak

# 2. パスフレーズなし鍵生成
ssh-keygen -t ed25519 -C "kinno.naoto.r6@dc.tohoku.ac.jp" -f ~/.ssh/id_ed25519_github -N ""

# 3. 公開鍵を表示（GitHub登録用）
cat ~/.ssh/id_ed25519_github.pub
```

#### 完全な設定手順
```bash
# SSH設定ファイル作成 (作成されている場合は必要なし)
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    IdentitiesOnly yes
EOF

# 権限設定
chmod 600 ~/.ssh/config ~/.ssh/id_ed25519_github
chmod 644 ~/.ssh/id_ed25519_github.pub

# SSH Agent設定
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github

# 接続テスト
ssh -T git@github.com
```

#### 成功の確認
以下のようなメッセージが表示されれば成功です：

```bash
# SSH接続テスト成功時
Hi kinno-7010! You've successfully authenticated, but GitHub does not provide shell access.

# プッシュ成功時
To github.com:kinno-7010/Research.git
   [commit1]..[commit2]  main -> main
```

### Claude Code環境での注意事項
1. **セキュリティ**: パスフレーズなしSSH鍵は、開発環境やClaude Code環境などの制限された環境でのみ使用してください
2. **権限管理**: SSH関連ファイルの権限設定を適切に行ってください
3. **定期更新**: セキュリティ向上のため、3-6ヶ月ごとに鍵の再生成を検討してください

### 環境特有の制約
- **対話的プロンプト不可**: パスフレーズやパスワードの入力ができません
- **GUI認証不可**: X11転送やGUIツールが利用できません
- **セッション制限**: SSH Agentの状態が維持されない場合があります

## 🚨 トラブルシューティング手順

### Step 1: 認証状態確認
```bash
# 現在の認証設定確認
git remote -v

# SSH認証の場合
ssh -T git@github.com

# Token認証の場合
echo "現在のトークン: ${GITHUB_TOKEN:0:10}..."
curl -s -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/user | jq '.login'
```

### Step 2: SSH認証エラー時の対処
```bash
# 1. SSH Agent再起動
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github

# 2. SSH接続テスト
ssh -T git@github.com

# 3. 権限確認
chmod 600 ~/.ssh/id_ed25519_github
chmod 644 ~/.ssh/id_ed25519_github.pub

# 4. SSH設定確認
cat ~/.ssh/config
```

### Step 3: Token認証エラー時の対処
```bash
# Bad credentials エラーの場合
echo "❌ トークンが無効です。新しいトークンを作成してください。"

# トークン再設定
read -s -p "新しいGitHubトークンを入力: " NEW_TOKEN
export GITHUB_TOKEN="${NEW_TOKEN}"
git remote set-url origin https://kinno-7010:${GITHUB_TOKEN}@github.com/kinno-7010/Research.git
```

### Step 4: 緊急時の手動プッシュ
```bash
# 直接認証情報を含むURL使用（一時的）
git push https://kinno-7010:[YOUR_TOKEN]@github.com/kinno-7010/Research.git main
```

## 🔄 定期メンテナンス

### 月次チェック項目
- [ ] SSH鍵の状態確認
- [ ] Personal Access Token の有効期限確認
- [ ] 認証接続テスト実行
- [ ] 大容量ファイルの除外設定確認
- [ ] ブランチ同期状況確認

### トークン更新手順
```bash
# 1. 有効期限チェック
curl -s -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/user

# 2. 新しいトークン作成（GitHub.com）
# 3. 環境変数更新
export GITHUB_TOKEN="新しいトークン"
echo 'export GITHUB_TOKEN="新しいトークン"' >> ~/.bashrc

# 4. リモートURL更新
git remote set-url origin https://kinno-7010:${GITHUB_TOKEN}@github.com/kinno-7010/Research.git
```

## 🆘 緊急対応プロトコル

### 認証完全失敗時
```bash
# 1. 現在の状態確認
git status
git log --oneline -3

# 2. ローカルコミット保護
git branch backup-$(date +%Y%m%d)

# 3. SSH認証に切り替え
git remote set-url origin git@github.com:kinno-7010/Research.git

# 4. Token認証に切り替え（SSH失敗時）
export GITHUB_TOKEN="新しいトークン"
git remote set-url origin https://kinno-7010:${GITHUB_TOKEN}@github.com/kinno-7010/Research.git
```

## 📝 重要な注意事項

1. **認証方式**: SSH認証を推奨、Token認証は代替手段
2. **トークンの機密性**: 絶対にコードにハードコードしない
3. **データファイル管理**: 大容量観測データは.gitignoreで除外
4. **コミット頻度**: 研究の節目ごとに適切にコミット
5. **ブランチ戦略**: 実験的解析は別ブランチで実施
6. **バックアップ**: 重要な解析結果は複数箇所で保管

### 自動化スクリプト例
```bash
#!/bin/bash
# quick_commit.sh - 研究用クイックコミット
TIMESTAMP=$(date '+%Y%m%d %H:%M')
git add **/*.py **/*.ipynb CLAUDE.md
git commit -m "${TIMESTAMP} : ${1:-研究進捗の更新}"
git push origin main
```

使用方法:
```bash
chmod +x quick_commit.sh
./quick_commit.sh "解析結果を更新"
```

