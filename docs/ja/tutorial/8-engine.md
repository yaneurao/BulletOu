# 8. エンジンに組み込む — やねうら王で動作確認

<a href="../../en/tutorial/8-engine.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習結果をやねうら王エンジンで動作確認する最小手順。

## 8.1 NNUE 系 (`nn.bin`)

最新の `000N/nn.bin` をエンジンが探す場所に置く。やねうら王の場合、`EvalDir` オプションでパスを指定する:

```
# エンジン起動後、USI コマンドで:
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/NNUE_HALFKP-256x2-32-32/0005
isready
bench
```

または、`eval/nn.bin` という相対パスでエンジン側に置く場合は、`000N/nn.bin` をそのファイル名で配置する。

`isready` でロードが通れば学習結果が認識できている。`bench` の出力に nn.bin のハッシュが出るので、毎回違う数字になっていれば確かに違う重みを load していることが分かる。

## 8.2 KPPT 系 (`KK_synthesized.bin` 等の 3 ファイル組)

最新 `000N/` ディレクトリそのものを `EvalDir` に指定する (3 ファイルが揃った状態のディレクトリを指す):

```
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/KPPT/0005
isready
bench
```

3 ファイルすべてが揃っていない場合エンジンは load に失敗する点に注意。

## 8.3 学習結果が弱いとき

最初の学習は小さな教師で短い superbatch しか回していないので、評価の質はあまり期待しないこと。本格対局できるレベルにするには:
- 教師サイズを増やす (1 億 → 10 億局面以上)
- `--max-epochs 3` 程度で複数周回す
- `--save-rate` を大きく (例: 10) して、後半の save だけを使う

詳細なハイパーパラメータ調整は各 eval-type のリファレンス ([halfkp.md](../shogi/halfkp.md) / [kp.md](../shogi/kp.md) / [halfkpe9.md](../shogi/halfkpe9.md) / [kppt.md](../shogi/kppt.md)) を参照。

---

前へ: [7. 結果を確認する](7-result.md)
