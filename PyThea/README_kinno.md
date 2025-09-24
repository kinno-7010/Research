# PyThea セットアップ & 起動メモ（WSL, Python 3.11 venv 用）
Documentation page: https://www.pythea.org/

**対象環境**
- WSL（Ubuntu）
- Python **3.11** の仮想環境：`~/venvs/pythea-311`
- 作業ディレクトリ：`/mnt/d/wsl/home/kinno-7010/Research`
- PyThea は **pip で導入済み**

---

## 1. 最短の起動手順

```bash
# 任意：作業ディレクトリへ
cd /mnt/d/wsl/home/kinno-7010/Research

# 仮想環境を有効化（Python 3.11）
source ~/venvs/pythea-311/bin/activate

# 参考：バージョン確認
which python
python -V
# /home/kinno-7010/venvs/pythea-311/bin/python
# Python 3.11.9

# PyThea を「直接起動」する（確実）
streamlit run "$(python -c 'import os, PyThea; print(os.path.dirname(PyThea.__file__))')/PyThea_app.py"
```
起動に成功すると、次のような表示が出ます：

```nginx
コードをコピーする
Local URL:   http://localhost:8501
Network URL: http://<WSL-IP>:8501
```
→ ブラウザで Local URL を開くと GUI が使えます。

WSL では gio: http://localhost:8501: Operation not supported が出ることがありますが、無視してOK。URL を手で開けば問題ありません。

終了はターミナルで Ctrl + C。

※ 初回だけ pip install --upgrade pip setuptools wheel && pip install PyThea が必要でした（すでに完了済み）。次回以降は不要です。

## 2. ショートカット（.bashrc 追記：任意）
一発切替や即起動をしたい場合、.bashrc の末尾に追記します。

```bash

# --- PyThea shortcuts ---
# venv 切替のみ
pythea_activate() {
  [ -n "$VIRTUAL_ENV" ] && deactivate
  source "$HOME/venvs/pythea-311/bin/activate"
}

# venv 切替 + 直起動
pythea_run() {
  pythea_activate
  streamlit run "$(python -c 'import os, PyThea; print(os.path.dirname(PyThea.__file__))')/PyThea_app.py"
}

# 使いやすいコマンド名
alias pythea-venv='pythea_activate'
alias pythea-run='pythea_run'
```
使い方

```bash

pythea-venv   # → venv 切替（プロンプトは (pythea-311)）
pythea-run    # → 切替してそのまま PyThea 起動
```

## 3. 起動時プロンプトの表示名について
現状は (pythea-311) と表示されます（このままで動作上問題なし）。

任意で表示名を (pythea-venv) にしたい場合は、起動関数で一時的に上書きするのが安全です。

```bash
# .bashrc の pythea_activate を次のように変更（既存 venv を壊さず安全）
pythea_activate() {
  [ -n "$VIRTUAL_ENV" ] && deactivate
  VIRTUAL_ENV_PROMPT="(pythea-venv) " source "$HOME/venvs/pythea-311/bin/activate"
}
```
備考：pyvenv.cfg の prompt = ... を編集しても、環境によっては即時反映されないことがあります。確実なのは上記の「関数内で上書き」方式です。

## 4. 「Command 'pyenv' not found …」の抑止（対応済み）
venv のときだけ pyenv のフックを無効化して、メッセージを抑止します。.bashrc の末尾に配置（※末尾に置くのがポイント）。

```bash
# --- Silence pyenv hook only while a Python venv is active ---
if [ -n "$VIRTUAL_ENV" ]; then
  _pyenv_virtualenv_hook() { :; }
  [ -n "$PROMPT_COMMAND" ] && PROMPT_COMMAND=${PROMPT_COMMAND//_pyenv_virtualenv_hook;};
fi
```
これで 自動アクティベートは残したまま、警告だけ出なくなります。

### 5. よくある落とし穴・メモ
venv は移動しない（非移動）。フォルダを動かすと内蔵の絶対パスが壊れます。
→ 別場所に置きたいときは 新規作成し、旧環境で pip freeze > req.txt → 新環境で pip install -r req.txt が安全。

/mnt/d/（NTFS）は I/O が遅め・権限周りが複雑になることがあります。
→ venv はホーム側（~/venvs/...）に置くのが安定。今回その構成になっています。

うまく起動しない時は 直起動がおすすめ（本書のコマンドが直起動）。

何か起きたら最小確認：

```bash
which python
python -V
python -c 'import os, PyThea; print(os.path.dirname(PyThea.__file__))'
```
## 6. 参考：PyThea の通常起動コマンド（補足）
PyThea が PATH に登録されている場合は以下でも起動できます（環境によって未登録のことがあるため、直起動の方が確実）。

``` bash
PyThea streamlit
```