"""データ収集から学習・評価までをまとめる MLRun / Kubeflow パイプライン定義。"""

import mlrun
from kfp import dsl


@dsl.pipeline(name="MLOps Bot Master Pipeline")
def kfpipeline(
    html_links: str,
    model_name: str,
    pretrained_tokenizer: str,
    pretrained_model: str,
    epochs: str,
    use_deepspeed: bool,
    tokenizer_class: str = "transformers.AutoTokenizer",
    model_class: str = "transformers.AutoModelForCausalLM",
):
    """
    HTML 収集、データ整形、LLM 学習、評価を順に実行するワークフロー。

    :param html_links: HTML URL 一覧ファイル
    :param model_name: 記録時のモデル名
    :param pretrained_tokenizer: 事前学習 tokenizer 名
    :param pretrained_model: 事前学習モデル名
    :param epochs: 学習エポック数
    :param use_deepspeed: DeepSpeed を利用するかどうか
    :param tokenizer_class: tokenizer クラス名
    :param model_class: モデルクラス名
    """
    # Get our project object:
    # 日本語訳: 現在の MLRun プロジェクトを取得する。
    project = mlrun.get_current_project()

    # Collect Dataset:
    # 日本語訳: URL 一覧から HTML を集め、テキストファイル群へ変換する。
    collect_dataset_run = mlrun.run_function(
        function="data-collecting",
        handler="collect_html_to_text_files",
        name="data-collection",
        params={"urls_file": html_links},
        returns=["html-as-text-files:path"],
    )

    # Dataset Preparation:
    # 日本語訳: 収集済みテキストを prompt 形式の学習データへ整形する。
    prepare_dataset_run = mlrun.run_function(
        function="data-preparing",
        handler="prepare_dataset",
        name="data-preparation",
        inputs={"source_dir": collect_dataset_run.outputs["html-as-text-files"]},
        returns=["html-data:dataset"],
    )

    # Training:
    # 日本語訳: プロジェクトへ登録済みの training 関数を利用する。
    project.get_function("training")

    training_run = mlrun.run_function(
        function="training",
        name="train",
        inputs={"dataset": prepare_dataset_run.outputs["html-data"]},
        params={
            "model_name": model_name,
            "pretrained_tokenizer": pretrained_tokenizer,
            "pretrained_model": pretrained_model,
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
            "TRAIN_num_train_epochs": epochs,
            "use_deepspeed": use_deepspeed,
        },
        handler="train",
        outputs=["model"],
    )

    # evaluation:
    # 日本語訳: 学習済みモデルと整形済みデータを使って perplexity 評価を実行する。
    mlrun.run_function(
        function="training",
        name="evaluate",
        params={
            "model_path": training_run.outputs["model"],
            "model_name": pretrained_model,
            "tokenizer_name": pretrained_tokenizer,
        },
        inputs={"data": prepare_dataset_run.outputs["html-data"]},
        handler="evaluate",
    )
