"""
OpenAIのWhisper API を使用して、音声ファイル(m4a, mp3, wav, etc.)を文字起こしする．
詳しくはREADME_transcribe_audio.mdを参照．
"""

import os
import math
import json
from pydub import AudioSegment
from openai import OpenAI


def split_audio(input_file: str, output_dir: str, chunk_length_min: int = 10):
    """
    音声ファイルを指定分数ごとに分割して保存する
    """
    audio = AudioSegment.from_file(input_file, format="m4a")
    chunk_length = chunk_length_min * 60 * 1000  # msに変換
    num_chunks = math.ceil(len(audio) / chunk_length)

    os.makedirs(output_dir, exist_ok=True)

    filepaths = []
    for i in range(num_chunks):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, len(audio))
        chunk = audio[start:end]
        output_path = os.path.join(output_dir, f"chunk_{i+1}.mp3")
        chunk.export(output_path, format="mp3")
        filepaths.append(output_path)
        print(f"✅ {output_path} を作成しました ({start/60000:.1f}分〜{end/60000:.1f}分)")

    print(f"\n🎉 分割完了！ {num_chunks} 個のファイルを {output_dir}/ に保存しました。")
    return filepaths


def transcribe_files(filepaths: list, api_key: str, output_file: str, language: str):
    """
    Whisper APIで分割ファイルを文字起こしして、1つのJSONにまとめる
    """
    client = OpenAI(api_key=api_key)
    results = []

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(f"⏳ Transcribing {filename}...")
        with open(filepath, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language=language
            )
            results.append(transcript.to_dict())

    # JSONに保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 全ての分割ファイルを処理しました → {output_file}")




if __name__ == "__main__":
    # 🎵 入力ファイル
    input_file = "... .m4a"

    # 📂 分割音声の保存先
    output_dir = "..."

    # 📄 出力JSONファイル
    output_file = "... .json"

    # 🔑 OpenAI APIキー（環境変数を推奨）
    api_key = "sk-proj-..."

    # 1️⃣ 音声を分割
    filepaths = split_audio(input_file, output_dir, chunk_length_min=10)

    # 2️⃣ Whisper APIで文字起こし
    transcribe_files(filepaths, api_key, output_file, language="ja")
    
    """
    language:
    - ja: 日本語
    - en: 英語
    - zh: 中国語
    - ko: 韓国語
    - fr: フランス語
    - de: ドイツ語
    - it: イタリア語
    """
