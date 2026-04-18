# Changelog

## 2026-04-18

- `tutorial.ipynb` の主要マークダウンセルに、日本語の補足説明を追記。
- `tutorial.ipynb` のコードセルに、処理意図が追いやすい日本語コメントを追加。
- `tutorial.ipynb` の既存英語コメントに対して、`日本語訳:` を追記。
- `tutorial.ipynb` を JSON として再読み込みし、Notebook 構造が壊れていないことを確認。
- `src/data_collection.py` に、HTML 取得と見出しマーキングの流れを説明する日本語コメントと補足 docstring を追加。
- `src/data_preprocess.py` に、prompt 生成と JSONL 変換の意図を説明する日本語コメントを追加。
- `src/serving.py` に、前処理・モデル読み込み・推論・毒性判定の流れを説明する日本語コメントを追加。
- `src/trainer.py` に、MLRun 連携・LoRA 学習・perplexity 評価の処理を追いやすくする日本語コメントを追加。
- `src/training_workflow.py` に、各パイプラインステップの役割を説明する日本語コメントと docstring を追加。
- `python -m compileall src` により、`src` 配下の構文確認を実施。
