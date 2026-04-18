"""HTML 記事を取得し、学習前処理しやすいテキストファイル群へ変換する処理。"""

import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag

ARTICLE_TOKEN = "Article: "
HEADER_TOKEN = "### Human: "


def normalize(s: str) -> str:
    """
    Remove newline and tab characters from string.
    文字列中の改行とタブを除去して、見出しを 1 行で扱いやすい形へそろえる。
    """
    return s.replace("\n", "").replace("\t", "")


def mark_header_tags(soup: BeautifulSoup):
    """
    Adding header token and article token prefixes to all headers in html, in order to parse the text later easily.
    後段で記事タイトルと見出しを識別しやすいよう、HTML 見出しへ専用プレフィックスを付ける。

    :param soup: BeautifulSoup object of the html file
    :param soup: HTML を解析した BeautifulSoup オブジェクト
    """
    # h1-h6 の見出しだけを対象にし、テキスト化したあとでも境界が分かるようタグ付けする。
    nodes = soup.find_all(re.compile("^h[1-6]$"))
    # Tagging headers in html to identify in text files:
    # 日本語訳: テキストファイル化した後でも見出しを識別できるよう、HTML の見出しへ印を付ける。
    if nodes:
        content_type = type(nodes[0].contents[0])
        # 最初の見出しは記事タイトルとして扱い、Article トークンを付与する。
        nodes[0].string = content_type(
            ARTICLE_TOKEN + normalize(str(nodes[0].contents[0]))
        )
        for node in nodes[1:]:
            if node.string:
                content_type = type(node.contents[0])
                if content_type == Tag:
                    # 子要素付き見出しでも正規化した見出しトークンを埋め込めるようにする。
                    node.string = HEADER_TOKEN + normalize(node.string)
                else:
                    node.string = content_type(HEADER_TOKEN + str(node.contents[0]))


def get_html_as_string(url: str, mark_headers: bool) -> str:
    """
    Retrieve text from html URL.
    HTML ページを取得し、必要であれば見出しマーキングを行ったうえでテキストへ変換する。

    :param url: html URL
    :param mark_headers: Whether to add article and header prefixes to headers to text
    :param url: HTML の取得元 URL
    :param mark_headers: 記事タイトルと見出しを識別するプレフィックスを付けるかどうか

    :returns: html text content
    :returns: HTML 本文をテキスト化した内容
    """
    # read html source:
    # 日本語訳: HTML ソースを読み込む。
    req = Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
    web_html_content = urlopen(req).read().decode("utf-8")
    soup = BeautifulSoup(web_html_content, features="html.parser")
    if mark_headers:
        mark_header_tags(soup)
    return soup.get_text()


def collect_html_to_text_files(urls_file: str, mark_headers=True) -> str:
    """
    Retrieve all html text content from URLs as text files.
    URL 一覧を順番に取得し、各ページ本文をテキストファイルとして保存する。

    :param urls_file: html URLs file
    :param mark_headers: Whether to add article and header prefixes to headers to text
    :param urls_file: HTML URL を 1 行ずつ記載したファイル
    :param mark_headers: 見出しプレフィックスを付与するかどうか

    :returns: the directory name that contains all the content text files.
    :returns: 保存先ディレクトリ名
    """
    directory = "html_as_text_files"
    os.makedirs(directory, exist_ok=True)
    # Writing html files as text files:
    # 日本語訳: HTML をテキストファイルとして書き出す。
    with open(urls_file, "r") as f:
        urls = f.readlines()
    for url in urls:
        url = url.replace("\n", "")
        page_name = Path(url).name
        # URL 末尾をファイル名に使い、各記事を個別テキストとして保存する。
        with open(f"{directory}/{page_name}.txt", "w") as f:
            f.write(get_html_as_string(url, mark_headers))
    return directory
