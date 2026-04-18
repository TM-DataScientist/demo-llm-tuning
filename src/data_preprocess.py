"""収集済みテキストを LLM 学習向けの prompt 形式データセットへ整形する処理。"""

import json
import os
import tempfile
from pathlib import Path

from datasets import load_dataset

ARTICLE_TOKEN = "Article: "
HEADER_TOKEN = "### Human: "
CONTENT_TOKEN = "### Assistant: "

DATA_FORMAT = """### Human: {} {}
### Assistant: {}"""
END_OF_ARTICLE = "Latest Posts"


def convert_textfile_to_data_with_prompts(txt_file: Path):
    """
    Formatting the html text content into prompt form.
    Each header-content in the article is an element in the generated list of prompts.
    記事テキストを「見出しごとに 1 件の prompt」となる形式へ分割する。

    :param txt_file: text content as a string with tokens of headers.
    :param txt_file: 見出しトークン付きテキストファイルのパス
    :returns: list of prompts
    :returns: 学習用 prompt 文字列のリスト
    """
    # Read file:
    # 日本語訳: ファイルを読み込む。
    with open(txt_file, "r") as f:
        lines = f.readlines()

    start = 0
    end = 0
    subject_idx = []
    data = []
    # Dividing text into header - paragraph prompts:
    # 日本語訳: テキストを「見出し - 本文」の prompt 単位へ分割する。
    for i, line in enumerate(lines):
        if not start and line.startswith(ARTICLE_TOKEN):
            start = i
        elif HEADER_TOKEN + END_OF_ARTICLE in line:
            end = i
            break
        if line.startswith(HEADER_TOKEN):
            subject_idx.append(i)

    # 記事タイトルから末尾マーカー直前までを、実際の本文範囲として切り出す。
    article_content = lines[start:end]
    subject_idx = [subject_i - start for subject_i in subject_idx]
    article_name = article_content[0].replace(ARTICLE_TOKEN, "")
    for i, subject in enumerate(subject_idx):
        if subject + 1 in subject_idx:
            continue
        subject_data = article_content[subject].replace(HEADER_TOKEN, "")
        if i + 1 == len(subject_idx):
            content_end = len(article_content)
        else:
            content_end = subject_idx[i + 1]
        content_limits = subject + 1, content_end
        # 記事名と見出し、本文を 1 つの instruction-response 形式へまとめる。
        data.append(
            DATA_FORMAT.format(
                article_name,
                subject_data,
                "".join(article_content[content_limits[0] : content_limits[1]]),
            )
        )
    return data


def prepare_dataset(source_dir: str):
    """
    Build the dataset from text files as a 'text: prompt' structure.
    テキストファイル群を読み込み、Hugging Face Datasets へ渡しやすい DataFrame に整形する。

    :param source_dir: the directory that contains all the text files.
    :param source_dir: テキストファイル群を含むディレクトリ

    :returns: A dataset with all the prompts inside
    :returns: `text` 列に prompt を持つ pandas.DataFrame
    """
    path_list = Path(source_dir).glob("./*.txt")
    data = []
    # Converting text files into data in our prompt format:
    # 日本語訳: テキストファイルを、このチュートリアル用の prompt 形式へ変換する。
    for path in path_list:
        data.extend(convert_textfile_to_data_with_prompts(path))
    data_dir = tempfile.mkdtemp()
    os.makedirs(data_dir, exist_ok=True)
    # Hugging Face Datasets が読める JSONL を一時ディレクトリへ書き出す。
    with open(data_dir + "/html_data.jsonl", "w", encoding="utf8") as f:
        for item in data:
            # 一部データに混ざる不要文字を除去してから保存する。
            f.write(
                json.dumps({"text": item.replace("ﾂ", "")}, ensure_ascii=False) + "\n"
            )
    return load_dataset(data_dir)["train"].to_pandas()
