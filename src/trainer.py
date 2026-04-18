"""Hugging Face Trainer を使った LLM fine-tuning と評価を行う処理。"""

import os
import shutil
import tempfile
import zipfile
from abc import ABC
from typing import Any, Dict, List

import mlrun
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from mlrun.artifacts.manager import Artifact, PlotlyArtifact
from mlrun.datastore import DataItem
from mlrun.execution import MLClientCtx
from mlrun.frameworks._common import CommonTypes, MLRunInterface
from mlrun.utils import create_class
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from plotly import graph_objects as go
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

DEEPSPEED_CONFIG = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
    "comms_logger": {
        "enabled": True,
        "verbose": False,
        "prof_all": True,
        "debug": False,
    },
}


# ----------------------from MLRUN--------------------------------
# 日本語訳: ここから先は MLRun 連携のための暫定実装。
class HFTrainerMLRunInterface(MLRunInterface, ABC):
    """
    This is temporary and will be built in mlrun 1.5.0
    Interface for adding MLRun features for tensorflow keras API.
    MLRun 1.5.0 に取り込まれる予定の暫定インターフェースで、
    Hugging Face Trainer に MLRun のログ機能を差し込むために使う。
    """

    # MLRuns context default name:
    # 日本語訳: MLRun コンテキストの既定名。
    DEFAULT_CONTEXT_NAME = "mlrun-huggingface"

    # Attributes to replace so the MLRun interface will be fully enabled.
    # 日本語訳: MLRun 連携のため差し替えるメソッド一覧。
    _REPLACED_METHODS = [
        "train",
        # "evaluate"
    ]

    @classmethod
    def add_interface(
        cls,
        obj: Trainer,
        restoration: CommonTypes.MLRunInterfaceRestorationType = None,
    ):
        super(HFTrainerMLRunInterface, cls).add_interface(
            obj=obj, restoration=restoration
        )

    @classmethod
    def mlrun_train(cls):
        def wrapper(self: Trainer, *args, **kwargs):
            # Restore the evaluation method as `train` will use it:
            # 日本語訳: `train` 内で評価が呼ばれるため、必要に応じて evaluate を復元する。
            # cls._restore_attribute(obj=self, attribute_name="evaluate")

            # Call the original fit method:
            # 日本語訳: 元の train メソッドを実行する。
            result = self.original_train(*args, **kwargs)

            # Replace the evaluation method again:
            # 日本語訳: 学習後に evaluate を再び差し替える。
            # cls._replace_function(obj=self, function_name="evaluate")

            return result

        return wrapper


class MLRunCallback(TrainerCallback):
    """
    This is temporary and will be built in mlrun 1.5.0
    Callback for collecting logs during training / evaluation of the `Trainer` API.
    学習・評価中に発生する metric を収集し、MLRun へ記録するコールバック。
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        model_name: str = "model",
        tag: str = "",
        labels: Dict[str, str] = None,
        extra_data: dict = None,
    ):
        super().__init__()

        # Store the configurations:
        # 日本語訳: ログ記録に必要な設定を保持する。
        self._context = (
            context
            if context is not None
            else mlrun.get_or_create_ctx("./mlrun-huggingface")
        )
        self._model_name = model_name
        self._tag = tag
        self._labels = labels
        self._extra_data = extra_data if extra_data is not None else {}

        # Set up the logging mode:
        # 日本語訳: エポックごとのステップと metric を蓄積する内部状態を初期化する。
        self._is_training = False
        self._steps: List[List[int]] = []
        self._metric_scores: Dict[str, List[float]] = {}
        self._artifacts: Dict[str, Artifact] = {}

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._steps.append([])

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        recent_logs = state.log_history[-1].copy()

        recent_logs.pop("epoch")
        current_step = int(recent_logs.pop("step"))
        if current_step not in self._steps[-1]:
            self._steps[-1].append(current_step)

        for metric_name, metric_score in recent_logs.items():
            if metric_name.startswith("train_"):
                if metric_name.split("train_")[1] not in self._metric_scores:
                    self._metric_scores[metric_name] = [metric_score]
                continue
            if metric_name not in self._metric_scores:
                self._metric_scores[metric_name] = []
            self._metric_scores[metric_name].append(metric_score)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._is_training = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

        if self._is_training:
            return

    def _log_metrics(self):
        # 最新 metric を MLRun へ結果として記録し、必要に応じて推移グラフも出す。
        for metric_name, metric_scores in self._metric_scores.items():
            self._context.log_result(key=metric_name, value=metric_scores[-1])
            if len(metric_scores) > 1:
                self._log_metric_plot(name=metric_name, scores=metric_scores)
        self._context.commit(completed=False)

    def _log_metric_plot(self, name: str, scores: List[float]):
        # Initialize a plotly figure:
        # 日本語訳: Plotly の図オブジェクトを初期化する。
        metric_figure = go.Figure()

        # Add titles:
        # 日本語訳: グラフタイトルと軸名を設定する。
        metric_figure.update_layout(
            title=name.capitalize().replace("_", " "),
            xaxis_title="Samples",
            yaxis_title="Scores",
        )

        # Draw:
        # 日本語訳: metric 推移を折れ線グラフとして描画する。
        metric_figure.add_trace(
            go.Scatter(x=np.arange(len(scores)), y=scores, mode="lines")
        )

        # Create the plotly artifact:
        # 日本語訳: Plotly グラフを MLRun アーティファクトとして保存する。
        artifact_name = f"{name}_plot"
        artifact = PlotlyArtifact(key=artifact_name, figure=metric_figure)
        self._artifacts[artifact_name] = self._context.log_artifact(artifact)


def apply_mlrun(
    trainer: transformers.Trainer,
    model_name: str = None,
    tag: str = "",
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    labels: Dict[str, str] = None,
    extra_data: dict = None,
    **kwargs,
):
    """
    This is temporary and will be built in mlrun 1.5.0
    Hugging Face Trainer に MLRun インターフェースとログコールバックを追加する。
    """
    # Get parameters defaults:
    # 日本語訳: コンテキストが未指定なら既定の MLRun コンテキストを使う。
    if context is None:
        context = mlrun.get_or_create_ctx(HFTrainerMLRunInterface.DEFAULT_CONTEXT_NAME)

    HFTrainerMLRunInterface.add_interface(obj=trainer)

    if auto_log:
        trainer.add_callback(
            MLRunCallback(
                context=context,
                model_name=model_name,
                tag=tag,
                labels=labels,
                extra_data=extra_data,
            )
        )


class KWArgsPrefixes:
    MODEL_CLASS = "CLASS_"
    FIT = "FIT_"
    TRAIN = "TRAIN_"
    PREDICT = "PREDICT_"
    DATA_COLLATOR = "DC_"


def _get_sub_dict_by_prefix(src: Dict, prefix_key: str) -> Dict[str, Any]:
    # MLRun コンテキストに混在する追加パラメータを、接頭辞ごとに切り出す。
    return {
        key.replace(prefix_key, ""): val
        for key, val in src.items()
        if key.startswith(prefix_key)
    }


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    学習対象パラメータ数と全体に対する割合を表示する。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train(
    context: MLClientCtx,
    dataset: DataItem = None,
    pretrained_tokenizer: str = None,
    pretrained_model: str = None,
    model_class: str = None,
    tokenizer_class: str = None,
    model_name: str = "huggingface-model",
    use_deepspeed: bool = True,
):
    """
    LoRA と量子化を使ってベース LLM を fine-tuning し、学習済みモデルを MLRun へ記録する。

    :param context: MLRun 実行コンテキスト
    :param dataset: 学習データ
    :param pretrained_tokenizer: 事前学習 tokenizer 名
    :param pretrained_model: 事前学習モデル名
    :param model_class: モデルクラスの完全修飾名
    :param tokenizer_class: tokenizer クラスの完全修飾名
    :param model_name: 記録時に使うモデル名
    :param use_deepspeed: DeepSpeed を利用するかどうか
    """
    torch.cuda.empty_cache()
    # deepspeed_config_json = None
    # 日本語訳: DeepSpeed 設定ファイルを動的生成する実装は現状コメントアウトされている。
    # if use_deepspeed:
    #     deepspeed_config_json = os.path.join(tempfile.mkdtemp(), "ds_config.json")
    #     with open(deepspeed_config_json, "w") as f:
    #         json.dump(DEEPSPEED_CONFIG, f)
    if tokenizer_class:
        tokenizer_class = create_class(tokenizer_class)
    else:
        tokenizer_class = AutoTokenizer

    # tokenizer を読み込み、EOS を pad token として使えるようにそろえる。
    tokenizer = tokenizer_class.from_pretrained(
        pretrained_tokenizer,
        model_max_length=512,
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_pandas(dataset.as_df())

    def preprocess_function(examples):
        # 各 prompt をトークン化して Trainer へ渡せる形式にする。
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = None

    # コンテキストパラメータから DataCollator 設定だけを抽出して反映する。
    data_collator_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.DATA_COLLATOR
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, **data_collator_kwargs
    )

    # Parsing kwargs:
    # 日本語訳: MLRun コンテキストに渡された追加パラメータを用途別に抽出する。
    train_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.TRAIN
    )
    # if use_deepspeed:
    # 日本語訳: DeepSpeed を有効化する場合は train 引数へ設定ファイルを渡す。
    #     train_kwargs["deepspeed"] = deepspeed_config_json
    model_class_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.MODEL_CLASS
    )
    # Loading our pretrained model:
    # 日本語訳: ベースモデル名を優先順位付きで決定する。
    model_class_kwargs["pretrained_model_name_or_path"] = (
        model_class_kwargs.get("pretrained_model_name_or_path") or pretrained_model
    )
    train_kwargs["hub_token"] = train_kwargs.get("hub_token") or pretrained_tokenizer
    if not model_class_kwargs["pretrained_model_name_or_path"]:
        raise mlrun.errors.MLRunRuntimeError(
            "Must provide pretrained_model name as "
            "function argument or in extra params"
        )
    # 4bit 量子化設定を用意し、大きなモデルでも学習しやすくする。
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = create_class(model_class).from_pretrained(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        **model_class_kwargs,
    )

    # gradient checkpointing と k-bit 学習向け前処理を有効化する。
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Preparing training arguments:
    # 日本語訳: 学習用の TrainingArguments を構築する。
    training_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        optim="paged_adamw_8bit",
        gradient_accumulation_steps=2,
        warmup_steps=5,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        **train_kwargs,
    )

    # LoRA の差分学習設定を定義し、ベースモデルへ適用する。
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # Trainer を構築し、tokenizer と data collator をひも付ける。
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    apply_mlrun(trainer, model_name=model_name)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
        # 日本語訳: 警告を抑えるため use_cache を無効化する。推論時は再度有効化してよい。
    )

    # Apply training with evaluation:
    # 日本語訳: 学習を実行する。
    context.logger.info(f"training '{model_name}'")
    trainer.train()

    temp_directory = tempfile.TemporaryDirectory().name
    trainer.save_model(temp_directory)

    # Zip the model directory:
    # 日本語訳: 保存したモデルディレクトリを zip 化する。
    shutil.make_archive(
        base_name="model",
        format="zip",
        root_dir=temp_directory,
    )

    # Log the model:
    # 日本語訳: zip 化した学習済みモデルを MLRun モデルとして記録する。
    context.log_model(
        key="model",
        db_key=model_name,
        model_file="model.zip",
        tag="",
        framework="Hugging Face",
    )


def evaluate(
    context,
    model_path,
    data: pd.DataFrame,
    model_name: str = None,
    tokenizer_name: str = None,
):
    """
    Evaluating the model using perplexity, for more information visit:
    https://huggingface.co/docs/transformers/perplexity
    Perplexity を用いて fine-tuning 後モデルを評価する。

    :param context: mlrun context
    :param model_path: path to the model directory
    :param data: the data to evaluate the model
    :param model_name: name of base model
    :param tokenizer_name: name of base tokenizer
    :param context: MLRun 実行コンテキスト
    :param model_path: 学習済みモデルアーティファクトのパス
    :param data: 評価用データ
    :param model_name: ベースモデル名
    :param tokenizer_name: ベース tokenizer 名
    """
    # Get the model artifact and file:
    # 日本語訳: モデルアーティファクトと zip ファイル本体を取得する。
    (
        model_file,
        model_artifact,
        extra_data,
    ) = mlrun.artifacts.get_model(model_path)

    # Read the name:
    # 日本語訳: アーティファクト上のモデル名を読み取る。
    _model_name = model_artifact.spec.db_key

    # Extract logged model files:
    # 日本語訳: ログ済み zip を展開してモデルディレクトリへ戻す。
    model_directory = os.path.join(os.path.dirname(model_file), _model_name)
    with zipfile.ZipFile(model_file, "r") as zip_file:
        zip_file.extractall(model_directory)

    # Loading the saved pretrained tokenizer and model:
    # 日本語訳: tokenizer とベースモデルを読み込み、その上に PEFT 重みを適用する。
    dataset = Dataset.from_pandas(data)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda:0", trust_remote_code=True, load_in_8bit=True
    )
    model = PeftModel.from_pretrained(model, model_directory)
    model.eval()
    encodings = tokenizer("\n\n".join(dataset["text"][:5]), return_tensors="pt")

    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    # 長い系列をスライディングウィンドウで走査し、負の対数尤度を集計する。
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        # 日本語訳: 最終ループでは stride と異なる長さになる場合がある。
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids.cuda(), labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            # 日本語訳: loss は有効ラベル上の平均 CrossEntropyLoss で計算される。
            # 日本語訳: モデル内部でラベルを 1 つ左へずらすため、実際に loss 対象となるのは trg_len - 1 個のラベル。
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    # 集計した負の対数尤度から perplexity を算出して記録する。
    context.log_result("perplexity", ppl)
