"""cshogi 経由で hcpe → psv 変換し、HcpeDataLoader との照合テスト用 psv ファイルを生成する。

- hcpe (38 byte/レコード) を読み
- 各 hcp を cshogi.Board.set_hcp() で局面復元
- cshogi.Board.to_psfen() で PackedSfen (32 byte) を取得
- score / bestMove16 / gameResult を組み合わせて PackedSfenValue (40 byte) を生成

出力: <hcpe>.psv (40 byte/レコード)

注: hcpe には gamePly が含まれないため、psv の gamePly フィールドは 0 で埋める。
これは BulletOu の HcpeDataLoader の挙動と同じ。
"""

import sys
import numpy as np
import cshogi


def hcpe_to_psv(hcpe_path: str, psv_path: str, count: int = 0, batch: int = 100_000) -> None:
    """count=0 のときは全件変換。それ以外は先頭 count 件のみ。

    変換結果は `batch` 件ずつストリーミングで `psv_path` に書き出すため、
    大規模ファイルでもメモリ使用量はバッチサイズ分に抑えられる。
    """
    import os

    hcpe_item = cshogi.HuffmanCodedPosAndEval.itemsize  # 38
    file_size = os.path.getsize(hcpe_path)
    total = file_size // hcpe_item
    if count > 0:
        total = min(total, count)
    print(f"converting {total} hcpe records from {hcpe_path}")

    board = cshogi.Board()
    written = 0
    with open(psv_path, "wb") as g:
        while written < total:
            n = min(batch, total - written)
            hcpes = np.fromfile(
                hcpe_path,
                dtype=cshogi.HuffmanCodedPosAndEval,
                count=n,
                offset=written * hcpe_item,
            )
            psvs = np.zeros(n, dtype=cshogi.PackedSfenValue)
            for i in range(n):
                h = hcpes[i]
                board.set_hcp(h["hcp"])
                board.to_psfen(psvs[i]["sfen"])
                psvs[i]["score"] = h["eval"]
                psvs[i]["move"] = h["bestMove16"].view(np.uint16)
                psvs[i]["gamePly"] = 0
                psvs[i]["game_result"] = h["gameResult"]
                psvs[i]["padding"] = 0
            g.write(psvs.tobytes())
            written += n
            if (written // batch) % 5 == 0:
                print(f"  converted {written} / {total}")

    print(f"wrote {written} psv records to {psv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: hcpe_to_psv.py <hcpe-in> <psv-out> [count]", file=sys.stderr)
        sys.exit(1)
    hcpe_in = sys.argv[1]
    psv_out = sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    hcpe_to_psv(hcpe_in, psv_out, count)
