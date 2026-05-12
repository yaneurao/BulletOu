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


def hcpe_to_psv(hcpe_path: str, psv_path: str, count: int = 0) -> None:
    """count=0 のときは全件変換。それ以外は先頭 count 件のみ。"""
    # 入力 hcpe
    if count > 0:
        hcpes = np.fromfile(hcpe_path, dtype=cshogi.HuffmanCodedPosAndEval, count=count)
    else:
        hcpes = np.fromfile(hcpe_path, dtype=cshogi.HuffmanCodedPosAndEval)
    print(f"read {len(hcpes)} hcpe records from {hcpe_path}")

    # 出力 psv
    psvs = np.zeros(len(hcpes), dtype=cshogi.PackedSfenValue)

    board = cshogi.Board()
    for i in range(len(hcpes)):
        h = hcpes[i]
        board.set_hcp(h["hcp"])
        # PackedSfenValue['sfen'] は (32,) shape の uint8 view。直接書き込む
        board.to_psfen(psvs[i]["sfen"])
        psvs[i]["score"] = h["eval"]
        # cshogi の hcpe.bestMove16 (i16) と psv.move (u16) は同じビット表現
        psvs[i]["move"] = h["bestMove16"].view(np.uint16)
        psvs[i]["gamePly"] = 0  # hcpe に存在しない情報
        psvs[i]["game_result"] = h["gameResult"]
        psvs[i]["padding"] = 0
        if (i + 1) % 10000 == 0:
            print(f"  converted {i + 1} / {len(hcpes)}")

    psvs.tofile(psv_path)
    print(f"wrote {len(psvs)} psv records to {psv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: hcpe_to_psv.py <hcpe-in> <psv-out> [count]", file=sys.stderr)
        sys.exit(1)
    hcpe_in = sys.argv[1]
    psv_out = sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    hcpe_to_psv(hcpe_in, psv_out, count)
