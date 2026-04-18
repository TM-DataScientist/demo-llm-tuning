"""LLM 推論用の Serving Graph 各ステップを定義するサービング実装。"""

import json
import os
import zipfile
from typing import Any, Dict

import evaluate
import mlrun.artifacts
import numpy as np
import torch
import transformers
from mlrun.serving.v2_serving import V2ModelServer
from peft import PeftModel

SUBJECT_MARK = "### Human: "
CONTENT_MARK = "\n### Assistant: "
PROMPT_FORMAT = SUBJECT_MARK + "{}" + CONTENT_MARK


def preprocess(request: dict) -> dict:
    """
    convert the request to the required structure for the predict function
    リクエストをサービングモデルが期待する `inputs` 形式へ変換する。

    :param request: A http request that contains the prompt
    :param request: `prompt` を含む HTTP リクエスト本文
    """
    # Read bytes:
    # 日本語訳: bytes で来た場合は JSON として読み直す。
    if isinstance(request, bytes):
        request = json.loads(request)

    # Get the prompt:
    # 日本語訳: リクエストから prompt を取り出す。
    prompt = request.pop("prompt")

    # Format the prompt as subject:
    # 日本語訳: 入力を学習時と同じ Human / Assistant 形式の prompt へ整形する。
    prompt = PROMPT_FORMAT.format(str(prompt))

    # Update the request and return:
    # 日本語訳: サービング関数が期待する `inputs` 配列の形に詰め替えて返す。
    request = {"inputs": [{"prompt": [prompt], **request}]}
    return request


class LLMModelServer(V2ModelServer):
    """
    This is temporary and will be built in mlrun 1.5.0
    MLRun 1.5.0 に内包予定の暫定 LLM サーバー実装。
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_class: str = "AutoModelForCausalLM",
        tokenizer_class: str = "AutoTokenizer",
        # model args:
        model_args: dict = None,
        # Load from MLRun args:
        model_path: str = None,
        # Load from hub args:
        model_name: str = None,
        tokenizer_name: str = None,
        # Deepspeed args:
        use_deepspeed: bool = False,
        n_gpus: int = 1,
        is_fp16: bool = True,
        # peft model:
        peft_model: str = None,
        # Inference args:
        **class_args,
    ):
        # Initialize the base server:
        # 日本語訳: 基底サーバーの初期化を行う。
        super(LLMModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            **class_args,
        )

        # Save class names:
        # 日本語訳: 利用する Transformers クラス名を保持する。
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class

        # Save hub loading parameters:
        # 日本語訳: Hugging Face Hub から読む際のモデル名を保持する。
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or self.model_name

        # Save load model arguments:
        # 日本語訳: モデル読み込み時の追加引数を保持する。
        self.model_args = model_args

        # Save deepspeed parameters:
        # 日本語訳: DeepSpeed 推論設定を保持する。
        self.use_deepspeed = use_deepspeed
        self.n_gpus = n_gpus
        self.is_fp16 = is_fp16

        # PEFT parameters:
        # 日本語訳: LoRA / PEFT 追加重みの情報を保持する。
        self.peft_model = peft_model

        # Prepare variables for future use:
        # 日本語訳: 後でロード結果を入れるためのプレースホルダーを初期化する。
        self.model = None
        self.tokenizer = None
        self._model_class = None
        self._tokenizer_class = None

    def load(self):
        # Get classes:
        # 日本語訳: 文字列で渡された Transformers クラスを実体へ解決する。
        self._model_class = getattr(transformers, self.model_class)
        self._tokenizer_class = getattr(transformers, self.tokenizer_class)

        # Load the model and tokenizer:
        # 日本語訳: MLRun アーティファクトまたは Hub からモデルと tokenizer を読み込む。
        if self.model_path:
            self._load_from_mlrun()
        else:
            self._load_from_hub()

        # Use deepspeed if needed:
        # 日本語訳: 必要なら DeepSpeed 推論エンジンでモデルを包む。
        if self.use_deepspeed:
            import deepspeed

            self.model = deepspeed.init_inference(
                model=self.model,
                mp_size=self.n_gpus,
                dtype=torch.float16 if self.is_fp16 else torch.float32,
                replace_method="auto",
                replace_with_kernel_inject=True,
            )
        if self.peft_model:
            # LoRA などの追加重みがある場合は、ベースモデルへ上書き適用する。
            self._load_peft_model()

    def _extract_model(self, url):
        # Get the model artifact and file:
        # 日本語訳: モデルアーティファクトと zip ファイルを取得する。
        (
            model_file,
            model_artifact,
            extra_data,
        ) = mlrun.artifacts.get_model(url)

        # Read the name:
        # 日本語訳: アーティファクト上のモデル名を取得する。
        model_name = model_artifact.spec.db_key

        # Extract logged model files:
        # 日本語訳: 保存済み zip を展開して実ファイル群へ戻す。
        model_directory = os.path.join(os.path.dirname(model_file), model_name)
        with zipfile.ZipFile(model_file, "r") as zip_file:
            zip_file.extractall(model_directory)
        return model_directory

    def _load_peft_model(self):
        # PEFT 重みだけを別アーティファクトから取り出し、ベースモデルへ重ねる。
        model_directory = self._extract_model(self.peft_model)
        self.model = PeftModel.from_pretrained(self.model, model_directory)
        self.model.eval()

    def _load_from_mlrun(self):
        # MLRun に記録済みの学習済みモデル一式をローカルへ展開して読み込む。
        model_directory = self._extract_model(self.model_path)

        # Loading the saved pretrained tokenizer and model:
        # 日本語訳: 保存済み tokenizer とモデルを読み込む。
        self.tokenizer = self._tokenizer_class.from_pretrained(model_directory)
        self.model = self._model_class.from_pretrained(
            model_directory, **self.model_args
        )

    def _load_from_hub(self):
        # Loading the pretrained tokenizer and model:
        # 日本語訳: Hugging Face Hub から事前学習済み tokenizer とモデルを読み込む。
        self.tokenizer = self._tokenizer_class.from_pretrained(
            self.tokenizer_name,
            model_max_length=512,
        )
        self.model = self._model_class.from_pretrained(
            self.model_name, **self.model_args
        )

    def predict(self, request: Dict[str, Any]) -> dict:
        # Get the inputs:
        # 日本語訳: 前処理済み request から prompt と生成パラメータを取り出す。
        kwargs = request["inputs"][0]
        prompt = kwargs.pop("prompt")[0]

        # Tokenize:
        # 日本語訳: prompt をトークン列へ変換する。
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if self.model.device.type == "cuda":
            inputs = inputs.cuda()

        # Get the pad token id:
        # 日本語訳: 生成時に使う pad token id を決める。
        pad_token_id = self.tokenizer.eos_token_id

        # Infer through the model:
        # 日本語訳: モデルでテキスト生成を実行する。
        output = self.model.generate(
            input_ids=inputs,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        # Detokenize:
        # 日本語訳: 生成トークン列を人が読める文字列へ戻す。
        prediction = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return {"prediction": prediction, "prompt": prompt}

    def explain(self, request: Dict) -> str:
        return f"LLM model server named {self.name}"


def postprocess(inputs: dict) -> dict:
    """
    Postprocessing the generated output of the model
    モデル出力から Assistant 部分だけを取り出し、後段へ渡しやすい形へ整える。
    """
    # Read the prediction:
    # 日本語訳: モデルの生出力を取得する。
    prediction = inputs["outputs"]["prediction"]

    # Look for a 'Content: ' mark to know the model found the subject, otherwise, it is probably garbage:
    # 日本語訳: Assistant マーカーが見つかればその後ろを回答本文として扱い、見つからなければ未整形出力として扱う。
    content_index = prediction.find(CONTENT_MARK)
    if content_index == -1:
        output = f"I'm not sure about it but I'll do my best: {prediction}"
    else:
        output = prediction[content_index + len(CONTENT_MARK) :]

    return {
        "inputs": [
            {"prediction": output.strip(), "prompt": inputs["outputs"]["prompt"]}
        ]
    }


class ToxicityClassifierModelServer(V2ModelServer):
    """
    model that checks if the text contain toxicity language.
    入力文と生成文に有害表現が含まれるかを判定するフィルター。
    """

    def __init__(self, context, name: str, threshold: float = 0.7, **class_args):
        # Initialize the base server:
        # 日本語訳: 基底サーバーを初期化する。
        super(ToxicityClassifierModelServer, self).__init__(
            context=context,
            name=name,
            model_path=None,
            **class_args,
        )

        # Store the threshold of toxicity:
        # 日本語訳: 毒性ありと判定する閾値を保持する。
        self.threshold = threshold

    def load(self):
        # Hugging Face Evaluate の toxicity 指標をロードする。
        self.model = evaluate.load("toxicity", module_type="measurement")

    def predict(self, inputs: Dict) -> str:
        # Read the user's input and model output:
        # 日本語訳: ユーザー入力とモデル出力を取り出す。
        prediction = inputs["inputs"][0]["prediction"]
        prompt = inputs["inputs"][0]["prompt"]

        # Infer through the evaluator model:
        # 日本語訳: 入力文と出力文の両方に対して toxicity を評価する。
        result = self.model.compute(predictions=[prediction, prompt])["toxicity"]
        if any(np.array(result) > self.threshold):
            return "This bot do not respond to toxicity."

        return prediction

    def explain(self, request: Dict) -> str:
        return f"Text toxicity classifier server named {self.name}"
