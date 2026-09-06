# RynnWorld-4D 環境セットアップ・オンボーディングガイド

**作成日**: 2026-09-06

**対象**: 新しいLLMエージェント、開発メンバー

**プロジェクト**: RynnWorld-4D (`yuki-inaho/RynnWorld-4D`)
**目的**: Blackwell対応ブランチの状態を安全に再確認し、RGB-DF学習・推論作業を再現可能な手順で引き継ぐ。

## 1. 最初に行うこと

新しいエージェントは、実装を始める前に次の順序で状態を確認する。

1. `README.md` と本書を最後まで読む。
2. `git status --short --branch` でブランチと未コミット差分を確認する。
3. `git fetch yuki blackwell` を実行し、`git rev-list --left-right --count HEAD...yuki/blackwell` が `0 0` か確認する。
4. 差分がある場合は所有者と意図を調べ、既存変更を上書きしない。
5. `pixi install`、`pixi run verify`、`pixi run test-cpu` を順に実行する。
6. GPU作業前に `pixi run verify-gpu` と `nvidia-smi` を確認する。
7. privateデータを使う場合だけ環境変数を設定し、`pixi run train-ready` でfail-closed確認する。
8. `outputs/`、`results/`、`training/`、`pretrained/` はローカル成果物として扱い、Gitへ追加しない。

## 2. プロジェクト概要

RynnWorld-4DはRGB、depth、optical flowを同期生成するロボット操作向け4D world modelである。本ブランチには、既存の3段階学習に加え、COLMAP RGB-DFデータをHydraで管理する学習経路、TensorBoard、Top-K checkpoint、AMUSE optimizer、compact checkpoint推論wrapperがある。

主要コンポーネント:

- `finetune_rynnworld4d.py`: RGB-DF world model fine-tuning本体
- `scripts/train_colmap_rgbdf.py`: Hydra管理のCOLMAP RGB-DF launcher
- `core/finetune/datasets/`: source/latent dataset契約
- `core/finetune/optim/`: AMUSE関連実装
- `core/finetune/topk_checkpointing.py`: Top-K保存
- `scripts/run_finetuned_rgbdf_inference.py`: compact checkpoint推論wrapper
- `rynnworld4d_policy/`: policy headとTianjiサンプル

## 3. 現在の状態

### 完了済み

| 分類 | 状態 | 内容 |
|---|---|---|
| Blackwell環境 | 完了 | Pixi環境、CUDA確認task、32GB GPU向けsmoke経路を整備 |
| 学習管理 | 完了 | Hydra、TensorBoard、Top-K、AMUSEを実装 |
| RGB-DFデータ | 完了 | source監査、materialize、latent encode、split検証を実装 |
| checkpoint | 完了 | compact checkpoint保存・読込・推論wrapperを実装 |
| Git | 完了 | `blackwell` が `yuki/blackwell` を追跡 |

### ローカルにしかないもの

- full COLMAP RGB-D/RGB-DFデータ
- ダウンロード済みpretrained model
- 学習checkpoint、TensorBoard event、生成動画
- `temp/` 以下の作業記録

これらは意図的に`.gitignore`対象であり、公開Gitへ入れない。

### 別データであることに注意

`data/tianji_sample/` と `data/tianji_sample_depth/` は追跡済みのTianji Pick-Placeサンプルである。COLMAPトマト温室RGB-Dとは別のデータであり、同一sourceとして扱わない。

## 4. 前提条件

```bash
git branch --show-current
git log --oneline -1
pixi --version
nvidia-smi
df -h .
```

期待する作業ブランチは`blackwell`、追跡先は`yuki/blackwell`である。GPU学習にはCUDA対応GPUが必要だが、`verify`と`test-cpu`はGPUなしで実行できる。

private COLMAP workflowでは次の環境変数が必須である。

```bash
export RYNNWORLD4D_COLMAP_ROOT=/path/to/private/colmap_rgbd
export RYNNWORLD4D_LATENT_ROOT=/path/to/local/colmap-rgbdf-artifacts
```

値そのものをMarkdown、commit message、ログの引用、issueへ記録しない。

## 5. 環境セットアップ

```bash
pixi install
pixi run verify
pixi run test-cpu
```

pretrained modelが必要で、十分な空き容量がある場合のみ次を実行する。

```bash
pixi run download-models
```

このtaskは複数の大規模modelを取得する。実行前にストレージ、利用規約、ネットワークを確認する。

## 6. 動作確認と学習順序

### CPU preflight

```bash
pixi run check-cpu
pixi run training-config
```

### privateデータ検証

```bash
pixi run audit-rgbdf-cpu
pixi run prepare-rgbdf-cpu
pixi run encode-rgbdf-latents
pixi run train-ready
```

既にmaterialize済みの場合でも、入力rootと生成先を確認してから実行する。既存artifactを暗黙に上書きしてはならない。

### GPU smokeからfullへ

```bash
pixi run verify-gpu
pixi run train-rgbdf-smoke
```

smokeでloss/gradient finite、checkpoint reload、GPU余裕を確認できた場合のみfullを開始する。

```bash
pixi run train-rgbdf-full
```

pretrained Stage-3の単一step確認は次を使う。

```bash
pixi run train-pretrained-amuse-smoke
```

TensorBoard:

```bash
export RYNNWORLD_TENSORBOARD_LOGDIR=outputs/training
pixi run tensorboard
```

## 7. 成果物の確認順序

1. Hydraのresolved configとoverrideを確認する。
2. TensorBoardでtrain/validation loss、learning rate、gradientを確認する。
3. leaderboardとTop-K checkpointのmetric、epoch、stepを照合する。
4. `last`をbestと誤認しない。
5. 推論wrapperでbest checkpointを読み、RGB/depth/flowのshapeとfiniteを検査する。
6. 生成動画は定量metricの代替にせず、代表フレームと時間方向の破綻確認に使う。

## 8. トラブルシューティング

### Hydraが環境変数未設定で停止する

仕様どおりである。private pathの暗黙fallbackは作らず、必要な環境変数を明示設定する。

### CUDA OOM

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
```

他processの使用量、batch、frame数、activation checkpoint、offload設定を一つずつ確認する。失敗runを上書きせず別runとして残す。

### compact checkpointが読めない

checkpointのformat、source commit、wrapperのkey正規化を確認する。strict loadを回避するための無条件`strict=False`は追加しない。

### 依存関係が壊れた

手動`pip install`を重ねず、まず`pixi install`と`pixi.lock`の差分を確認する。

## 9. 次のステップ

優先順位:

1. remoteとworktreeがcleanであることを確認する。
2. private artifactの所在とSHA-256をlocal recordで照合する。
3. CPU testとGPU smokeを再実行する。
4. best checkpointを使った再現可能な推論を行う。
5. 新しい性能主張には同一split・同一集計方法の比較reportを付ける。
6. commit前に`git diff --cached`とsecret/path scanを行う。

## 10. 完了チェックリスト

- [ ] `README.md`と本書を読んだ
- [ ] `blackwell`と`yuki/blackwell`の同期を確認した
- [ ] 既存差分の所有者と意図を確認した
- [ ] `pixi install`と`pixi run test-cpu`が成功した
- [ ] GPU作業前に利用可能メモリを確認した
- [ ] private rootを環境変数だけで渡した
- [ ] smokeを通してからfullを開始した
- [ ] best/last/Top-Kを区別した
- [ ] data、weights、logs、outputsをstageしていない
- [ ] commit前にstaged diffとremote divergenceを再確認した

## 11. 更新履歴

- 2026-09-06 UTC: Blackwell、Hydra RGB-DF、AMUSE、checkpoint推論の現状に合わせて初版作成。
