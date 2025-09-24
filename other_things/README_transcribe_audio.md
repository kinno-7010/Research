# README_transcribe_audio.md

2025/09/21 文責：金野直人 (理学研究科地球物理学専攻修士2年) & ChatGPT 5

## 📌 概要
このプログラムは **OpenAI Whisper API** を使って、合唱や講習会などの音声ファイルを文字に変換（文字起こし）するためのものです。  
音声が長い場合（例：50分の録音）、自動的に10分ごとに分割し、それぞれを文字起こししたあとで **ひとつのJSONファイルにまとめます**。  

「先生がどこで何を言ったか」を記録して、練習の復習に役立てることができます。  

このPythonコードは主にAIを使った文字起こしのためのコードになっています．
文字起こししたJSONファイルの内容をまとめるためには，ChatGPTやGeminiなどの他のAIに「～という内容でまとめてください」とプロンプトを入力してください．
---

## 🛠️ 使うまでの準備

### 1. OpenAI APIキーの取得
1. [OpenAI公式サイト](https://platform.openai.com/) にアクセスしてアカウントを作成します。  
2. [API Keys](https://platform.openai.com/account/api-keys) ページから新しいAPIキーを発行します。  
   → `sk-xxxxxxxx` のような文字列が発行されます。  
3. セキュリティのため、プログラムに直接書かず **環境変数に保存** してください。  

Linux/WSL/macOS の場合：
```bash
export OPENAI_API_KEY="sk-xxxxxxxx"
```

Windows (PowerShell) の場合：
```powershell
setx OPENAI_API_KEY "sk-xxxxxxxx"
```

---

### 2. 音声ファイルの準備
- 使える形式は `m4a`, `mp3`, `wav` です。  
- 文字起こししたい音声ファイルをこのプロジェクトのフォルダに入れてください。  

例：
```
project/
 ├── transcribe_audio.py
 ├── Geistesgruss_voice.m4a
 └── output/
```
#### 動画から音声ファイルにする方法
- Microsoft Clipchamp などをインストール
- 「新しい動画を編集」→「メディアのインポート」で動画を入れる
- メディアをドラッグ & ドロップ
- エクスポート→音声のみ　でm4aファイルがダウンロードされる．適切なディレクトリにm4aファイルを移動する．

---

### 3. 必要なライブラリのインストール
ターミナルで以下を実行してください：
```bash
pip install openai pydub
```

さらに `pydub` の動作に必要な **ffmpeg** をインストールします。  

- Ubuntu/WSL:  
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```
- macOS (Homebrew):  
  ```bash
  brew install ffmpeg
  ```
- Windows: [ffmpeg公式サイト](https://ffmpeg.org/download.html) からダウンロードして PATH を設定してください。

---

## ▶️ 実行方法

```bash
python3 transcribe_audio.py
```

実行すると：
1. 音声ファイルを10分ごとに分割（mp3形式）  
2. Whisper APIに送信して文字起こし  
3. `transcript_full.json` に結果を保存  

---

## 📄 プログラムの説明

### 1. `split_audio(input_file, output_dir, chunk_length_min=10)`
- 音声ファイルを指定した長さ（デフォルトは10分）で分割します。  
- 出力は軽量な `mp3` 形式です。  

### 2. `transcribe_files(filepaths, api_key, output_file)`
- 分割した音声を Whisper API (`whisper-1`) に送って文字起こしします。  
- 結果は `verbose_json` 形式で、時間ごとのセグメント情報も含まれます。  
- すべての結果を1つのJSONにまとめます。  

### 3. 実行部分 (`__main__`)
- 入力ファイル、出力フォルダ、APIキーを設定します。  
- 音声分割 → 文字起こし → JSON保存 の処理を自動で実行します。  

---

## 📂 出力結果

- `chunks/` フォルダに分割された音声ファイル  
  ```
  chunk_1.mp3, chunk_2.mp3, ...
  ```
- `transcript_full.json` にすべての文字起こし結果  

出力イメージ：
```json
[
  {
    "text": "はい、そこはもう少し柔らかく歌ってください。",
    "segments": [
      {"start": 5.0, "end": 9.2, "text": "はい、そこはもう少し柔らかく歌ってください"}
    ]
  },
  {
    "text": "ドイツ語の母音を意識して発音しましょう。",
    "segments": [
      {"start": 15.3, "end": 20.1, "text": "ドイツ語の母音を意識して発音しましょう"}
    ]
  }
]
```

---

## 💰 Whisper APIの料金目安
Whisper API は **従量課金制** です。  

- 料金：**0.006ドル / 1分**  
- 例：  
  - 10分 → 約1円弱  
  - 50分 → 約45円  
  - 1時間 → 約54円  

つまり、合唱の練習や講習会の録音を文字起こししても **数十円程度** で済みます。  

---

## ✅ 応用
- `chunk_length_min` を5にすれば、5分ごとに分割可能です。  
- 出力を整形すれば「先生の注意集」や「練習チェックリスト」に加工できます。  

---

👉 これで合唱部の練習や講習会を記録して、効率的に復習できるようになります。  
