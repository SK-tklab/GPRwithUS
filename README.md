# GPRwithUS
Gaussian process regression with uncertainty sampling

## 概要
ガウス過程回帰をnumpyのみで実装
ガウス過程の詳細については以下を参照
GPML http://www.gaussianprocess.org/gpml/chapters/RW.pdf

Uncertainty samplingは能動学習でモデルの精度を効率的に向上させるための方法の一つで, 分散最大となる点を次の観測点とする.

実行すると回帰結果のプロットが保存される


## 設定詳細
- 観測ノイズ: 1e-4
- カーネル: RBFカーネル
  - length scale = 0.4
  - variance = 1
