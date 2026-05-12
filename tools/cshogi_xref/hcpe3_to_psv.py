"""cshogi 経由で hcpe3 → psv 変換し、Hcpe3DataLoader との照合テスト用 psv ファイルを生成する。

hcpe3 (HuffmanCodedPosAndEval3) は dlshogi 系のゲーム単位可変長フォーマット:

    [ HuffmanCodedPosAndEval3 (36 byte) ]
       hcp: u1[32]
       moveNum: u16
       result: u8        ← 下位 2bit が勝敗 (0=Draw 1=BlackWin 2=WhiteWin)、他は特殊終了フラグ
       gameInfo: u8

    moveNum 回繰り返し:
        [ MoveInfo (6 byte) ]
           selectedMove16: i16  (cshogi 形式 move16)
           eval: i16
           candidateNum: u16

        candidateNum 回繰り返し:
            [ MoveVisits (4 byte) ]
               move16: i16
               visitNum: u16

このスクリプトは、各ゲームを moveNum 個の局面に展開し、それぞれの局面を
cshogi.Board.to_psfen() で PackedSfen 化して、PackedSfenValue (40 byte) として書き出す。

各 PackedSfenValue のフィールド:
    sfen        = cshogi.Board.to_psfen() (32 byte)
    score       = MoveInfo[i].eval
    move        = MoveInfo[i].selectedMove16 (i16 ビットを u16 として保存)
    gamePly     = i (0-indexed ply within the game; matches BulletOu's
                     MiniPosition.game_ply progression: from_hcp sets ply=0,
                     do_move increments by 1)
    game_result = header.result & 0x3 を i8 として保存
    padding     = 0

MoveVisits は読み飛ばす (本スクリプトは value 学習用の psv 生成専用)。
"""

import struct
import sys
import numpy as np
import cshogi


# --- hcpe3 dtype 定義 (dlshogi 系) ---

HUFFMAN_CODED_POS_AND_EVAL3 = np.dtype(
    [
        ("hcp", "u1", (32,)),
        ("moveNum", np.uint16),
        ("result", np.uint8),
        ("gameInfo", np.uint8),
    ]
)

MOVE_INFO = np.dtype(
    [
        ("selectedMove16", np.int16),
        ("eval", np.int16),
        ("candidateNum", np.uint16),
    ]
)

MOVE_VISITS = np.dtype(
    [
        ("move16", np.int16),
        ("visitNum", np.uint16),
    ]
)


def hcpe3_to_psv(hcpe3_path: str, psv_path: str, max_games: int = 0, max_positions: int = 0) -> None:
    """hcpe3 → psv 変換。

    max_games > 0 のときはその数のゲームで打ち切り (0=全件)。
    max_positions > 0 のときはその数の局面で打ち切り (0=全件)。

    psv は局面ごとにストリーミング書き出しするため、メモリ使用量はゲーム数によらず
    O(1) で済む。
    """
    games_read = 0
    positions_written = 0
    truncated = False

    board = cshogi.Board()

    with open(hcpe3_path, "rb") as f, open(psv_path, "wb") as g:
        while True:
            header_bytes = f.read(HUFFMAN_CODED_POS_AND_EVAL3.itemsize)
            if len(header_bytes) < HUFFMAN_CODED_POS_AND_EVAL3.itemsize:
                break  # EOF
            header = np.frombuffer(header_bytes, dtype=HUFFMAN_CODED_POS_AND_EVAL3, count=1)[0]
            move_num = int(header["moveNum"])
            # 0/1/2 は i8 でも u8 でも同じビット表現
            game_result_i8 = int(header["result"]) & 0x3

            # 開始局面をセット
            board.set_hcp(header["hcp"])

            for i in range(move_num):
                mi_bytes = f.read(MOVE_INFO.itemsize)
                if len(mi_bytes) < MOVE_INFO.itemsize:
                    print(f"truncated MoveInfo at game {games_read} ply {i}", file=sys.stderr)
                    truncated = True
                    break
                mi = np.frombuffer(mi_bytes, dtype=MOVE_INFO, count=1)[0]
                cand_num = int(mi["candidateNum"])

                # 現在の局面 (i 手目を指す前) を PSV 化
                psfen = np.zeros(32, dtype=np.uint8)
                board.to_psfen(psfen)

                # PackedSfenValue (40 byte) を組み立てて即書き出し
                move16_u16 = int(mi["selectedMove16"]) & 0xFFFF
                score_i16 = int(mi["eval"])
                psv_bytes = bytearray(40)
                psv_bytes[0:32] = psfen.tobytes()
                psv_bytes[32:34] = struct.pack("<h", score_i16)
                psv_bytes[34:36] = struct.pack("<H", move16_u16)
                psv_bytes[36:38] = struct.pack("<H", i)  # gamePly = ply within game (0-indexed)
                psv_bytes[38] = game_result_i8 & 0xFF
                psv_bytes[39] = 0  # padding
                g.write(psv_bytes)
                positions_written += 1

                # MoveVisits を読み飛ばし
                if cand_num > 0:
                    skip = MOVE_VISITS.itemsize * cand_num
                    skipped = f.read(skip)
                    if len(skipped) < skip:
                        print(f"truncated MoveVisits at game {games_read} ply {i}", file=sys.stderr)
                        truncated = True
                        break

                # 次の ply へ
                if i + 1 < move_num:
                    board.push_move16(int(mi["selectedMove16"]) & 0xFFFF)

                if max_positions > 0 and positions_written >= max_positions:
                    break

            games_read += 1
            if (games_read % 5000) == 0:
                print(f"  read {games_read} games, {positions_written} positions")

            if truncated:
                break
            if max_games > 0 and games_read >= max_games:
                break
            if max_positions > 0 and positions_written >= max_positions:
                break

    print(f"read {games_read} games -> wrote {positions_written} psv records to {psv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: hcpe3_to_psv.py <hcpe3-in> <psv-out> [max_games] [max_positions]",
            file=sys.stderr,
        )
        sys.exit(1)
    hcpe3_in = sys.argv[1]
    psv_out = sys.argv[2]
    max_games = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    max_positions = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    hcpe3_to_psv(hcpe3_in, psv_out, max_games, max_positions)
