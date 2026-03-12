# 小説本文テキスト使用アスキーアートノード 仕様書案

## 1. 目的

既存のカラーアスキーアートノードを拡張し、入力画像をアスキーアート化する際に、任意の `txt` ファイルに含まれる文章をそのまま文字列ソースとして使用する。

想定ユースケースは以下。

- ある小説のシーンを表す画像を生成する
- その画像をアスキーアート化する
- 使用文字は任意の文字セットではなく、その小説本文そのものを順番に消費する
- 文字配置は横書き、左から右、上から下とする

加えて、生成前に以下を把握できるようにする。

- 何文字必要か
- 現在の画像解像度とパラメータで何文字使われるか
- フォントサイズと出力解像度の関係
- 指定テキストと指定フォントで実際に描画可能か


## 2. 背景と前提

既存実装には以下の土台がある。

- `pixelation.py` による画像のグリッド化
- `ascii_art_custom_font.py` による高解像度キャンバスへの文字描画
- `ascii_utils.py` によるテキストファイル読み込み系ユーティリティの実装パターン

今回の要件は、既存の「輝度に応じて文字集合から文字を選ぶ」方式とは異なり、「文章を順番に消費する」方式である。そのため、単一ノードに全部詰め込むより、以下の 2 ノード構成が適切。

- 文字数・解像度・グリッドを事前確認する補助ノード
- 実際に本文文字列で描画する本体ノード


## 3. 提案ノード構成

### 3.1 ノードA: `ASCIITextLayoutPlanner`

目的:
画像サイズ、ピクセル化設定、フォント設定、本文ファイルから、必要文字数と描画成立性を事前に見積もる。

### 3.2 ノードB: `ASCIINovelTextArt`

目的:
実画像と本文テキストを使って、横書きの本文アスキーアートを生成する。


## 4. ノードA: `ASCIITextLayoutPlanner` 仕様

### 4.1 入力

- `image`
- `pixel_size`
- `aspect_ratio_correction`
- `resolution_scale`
- `font_name`
- `font_size`
- `text_file_path`
- `text_encoding`
  - 初期値: `utf-8`
  - 候補: `utf-8`, `utf-8-sig`, `shift_jis`, `cp932`, `auto`
- `newline_mode`
  - 初期値: `remove`
  - 候補: `remove`, `space`, `preserve`
- `text_shortage_mode`
  - 初期値: `error`
  - 候補: `error`, `loop`, `truncate_blank`
- `glyph_check_limit`
  - 初期値: `5000`
  - 用途: フォント描画可否チェックの最大文字数

### 4.2 出力

- `required_chars` : 実描画に必要な文字数
- `usable_chars` : 前処理後に実際に使える文字数
- `grid_width`
- `grid_height`
- `render_width`
- `render_height`
- `glyph_supported_ratio`
- `unsupported_chars_preview`
- `report_text`

### 4.3 文字数見積もりロジック

既存 `pixelation.py` のロジックに合わせる。

- `grid_width = max(1, image_width // pixel_size)`
- `effective_pixel_height = pixel_size * aspect_ratio_correction`
- `grid_height = max(1, int(image_height / effective_pixel_height))`
- `required_chars = grid_width * grid_height`

高解像度描画時の出力サイズは以下。

- `render_width = int(image_width * resolution_scale)`
- `render_height = int(image_height * resolution_scale)`

### 4.4 テキスト前処理

本文ファイル読み込み後、以下を適用する。

- 改行処理
  - `remove`: 改行コードを削除
  - `space`: 改行を半角スペースに置換
  - `preserve`: 改行を制御文字として保持
- タブは半角スペースに置換
- BOM は除去
- 前後の不要な空白は必要最小限のみ整形

初期仕様では、本文を「連続した文字列ストリーム」として扱うため、推奨初期値は `newline_mode=remove` とする。

### 4.5 フォント互換性チェック

Planner は、指定フォントで本文文字が描画可能かを事前確認する。

チェック内容:

- 指定フォントで `font.getbbox(char)` が取得できるか
- 空グリフが連続していないか
- サンプル範囲内で未対応文字を収集する

出力例:

- `glyph_supported_ratio = 0.97`
- `unsupported_chars_preview = "髙, 𠮟, …"`

備考:

- 日本語本文を扱うため、初期推奨フォントは `NotoSansJP-VariableFont_wght.ttf`
- 未対応文字が多い場合は、本描画前に Planner 側で警告を出す


## 5. ノードB: `ASCIINovelTextArt` 仕様

### 5.1 入力

- `image`
- `pixel_size`
- `resolution_scale`
- `downscale_mode`
- `aspect_ratio_correction`
- `font_name`
- `font_size`
- `text_file_path`
- `text_encoding`
- `newline_mode`
- `text_shortage_mode`
  - `error`: 必要文字数に足りなければエラー
  - `loop`: 先頭に戻って繰り返す
  - `truncate_blank`: 足りないセルは空白で埋める
- `char_color_mode`
  - 初期値: `sampled_color`
  - 候補: `sampled_color`, `grayscale`, `black`, `knockout_white`
- `background_mode`
  - 初期値: `white`
  - 候補: `white`, `transparent`, `sampled_average`, `source_image`
- `brightness`
- `contrast`
- `sharpen_mode`
  - 既存互換のため `None`, `unsharp`
- `sharpen_amount`
- `sharpen_threshold`
- `seed`

### 5.2 出力

- `output_image`
- `used_chars`
- `required_chars`
- `report_text`

### 5.3 描画方式

1. 入力画像を既存の `pixelation.py` ベースでグリッド化する
2. グリッドの各セルに対して、本文文字列を先頭から 1 文字ずつ割り当てる
3. 配置順は固定で、左から右、上から下
4. 描画モードに応じて、各セルの代表色または背景画像を使って文字表現を作る
5. 文字描画は `ascii_art_custom_font.py` の高解像度方式を流用する

### 5.3.1 描画モード

本体ノードは少なくとも以下の 2 パターンに対応する。

- 通常カラー文字モード
  - 既存ノードに近い方式
  - 各セルの代表色で文字を描画する
  - `char_color_mode=sampled_color|grayscale|black`
- 白抜きノックアウトモード
  - 背景には元画像またはその高解像度版を敷く
  - 文字形状の領域だけを白で置換する
  - 見た目としては「背景画像の上に白抜き文字を載せる」形になる
  - `char_color_mode=knockout_white` を使用する

初期仕様では、白抜きモードは「文字部分だけ背景画像から切り抜いて白にする」挙動として定義する。

### 5.3.2 白抜きノックアウトモードの処理

想定処理順:

1. 元画像を `resolution_scale` に応じて高解像度化した背景キャンバスを作る
2. 各セルに本文文字を割り当てる
3. 各文字のグリフマスクを生成する
4. そのマスク領域を背景画像上で白に置換する

実装上は以下のいずれかで対応可能。

- 背景キャンバスに対して、グリフマスクを使って白色を `paste` する
- 文字マスクを別レイヤーに蓄積し、最後に白レイヤーと合成する

初期実装では、既存の `ascii_art_custom_font.py` と整合しやすい前者を推奨する。

### 5.4 横書きルール

本仕様では、横書きのみ対応する。

- 1 行あたり `grid_width` 文字
- 行数は `grid_height`
- セル割当順は `y=0..grid_height-1`, `x=0..grid_width-1`
- 文章中の改行を `preserve` にした場合のみ、改行を明示的な行送りとして扱う余地を残す

ただし初期実装では複雑化を避けるため、`preserve` は Planner で情報保持のみ行い、本体ノードでは `remove` または `space` を推奨とする。

### 5.5 文字サイズ方針

初期仕様では、本文の読みやすさを優先して `font_size` は単一値とする。

理由:

- 文字ごとに大きくサイズが変わると、本文由来の文字列として視認しにくくなる
- 今回は「どの文字を置くか」が本文順で固定されるため、既存ノードほど強いサイズ変調の必要性がない
- 可読性は高解像度描画で確保しやすい

将来拡張:

- `font_size_min` / `font_size_max` の再導入
- 輝度連動で 10% 程度だけサイズ変動させる軽微モード

### 5.6 文字鮮明化

`ascii_art_custom_font.py` の以下の考え方を流用する。

- 高解像度キャンバスへ描画してから出力する
- `resolution_scale` を導入する
- 文字ごとのグリフマスクをキャッシュする
- `ImageDraw.text` ではなく、グリフのマスク画像を `paste` して輪郭を安定化する

この方式により、同じセル数でも文字の潰れを抑えやすい。

白抜きノックアウトモードでも同じグリフマスクを使うことで、通常文字モードと同等の輪郭品質を確保する。

### 5.6.1 背景の扱い

白抜きノックアウトモードでは、`background_mode=source_image` を事実上の標準とする。

理由:

- 「背景画像から切り抜く」という意図に最も忠実
- 通常カラー文字モードと違って、背景が白だと白抜き文字が消える

補足:

- `background_mode=white` と `char_color_mode=knockout_white` の組み合わせは視認性がないため、Planner もしくは本体ノードで警告対象にする
- `background_mode=transparent` は、透明背景上に白文字のみ残す用途としては有効

### 5.7 テキスト不足時の挙動

- `error`
  - 本文が不足したら明示的に停止
  - 事前見積もりとの整合が取りやすい
- `loop`
  - 本文先頭に戻って再利用
  - 長文テクスチャ的な表現向け
- `truncate_blank`
  - 残りセルを空白で埋める
  - 文字数不足が画像の欠けとして分かりやすい

初期推奨は `error`。


## 6. 推奨する実装判断

### 6.1 V1 ではやらないこと

- 縦書き
- 複数フォントの自動フォールバック
- 本文中のルビ、注記、外字の高度な解釈
- セルごとの複雑な字間・カーニング補正
- 改行 `preserve` の厳密な段落レイアウト再現

### 6.2 V1 でやるべきこと

- `txt` ファイル読込
- 文字数見積もり
- フォント対応可否の事前チェック
- 高解像度文字描画
- 横書き固定の本文順配置
- テキスト不足時の制御
- 通常カラー文字モード
- 白抜きノックアウトモード


## 7. データフロー

1. ユーザーが元画像を用意する
2. ユーザーが小説本文の `txt` を用意する
3. `ASCIITextLayoutPlanner` で必要文字数とフォント互換性を確認する
4. 問題なければ `ASCIINovelTextArt` に同じ設定を渡して描画する
5. 必要に応じて `pixel_size`, `resolution_scale`, `font_size` を再調整する


## 8. 実装メモ

### 8.1 既存コード再利用候補

- グリッド計算: `pixelation.py`
- 高解像度描画: `ascii_art_custom_font.py`
- テキスト読込ユーティリティの実装パターン: `ascii_utils.py`
- ノード登録: `__init__.py`

### 8.2 追加候補ユーティリティ

新規追加候補:

- `text_utils.py`

想定関数:

- `load_text_file(path, encoding, newline_mode) -> str`
- `normalize_text_stream(text) -> str`
- `count_renderable_chars(text, font_path, font_size, limit) -> dict`
- `iter_text_stream(text, shortage_mode) -> iterator`
- `apply_text_knockout(base_image, glyph_mask, position, fill_color=(255,255,255)) -> Image`


## 9. 受け入れ条件

- Planner の `required_chars` が、本体ノードで実際に必要となるセル数と一致する
- `resolution_scale=4.0` 以上で、既存標準ノードより文字輪郭が明確になる
- `text_shortage_mode=error` のとき、本文不足が無言で握り潰されない
- 指定フォント非対応文字がある場合、未対応文字の一部がレポートに出る
- 本文の配置順が常に左から右、上から下で再現可能
- `char_color_mode=knockout_white` で、文字形状の領域のみが白に置換される
- `background_mode=source_image` と組み合わせたとき、背景画像の情報を保ったまま白抜き文字表現になる


## 10. 最初の実装順

1. `ASCIITextLayoutPlanner` を実装する
2. `txt` 読込と文字正規化ユーティリティを追加する
3. `ASCIINovelTextArt` を固定 `font_size` 版で実装する
4. `ascii_art_custom_font.py` 相当の高解像度描画を統合する
5. 必要なら後続で可変フォントサイズや改行保持を追加する


## 11. この構成を推す理由

- 事前に必要文字数を把握したい、という要件を Planner が直接満たせる
- 本文由来の文字列を使う場合、フォント非対応文字が実運用上の大きな失敗要因になるため、事前チェックを独立させる価値が高い
- `ascii_art_custom_font.py` の高解像度描画を流用しやすい
- 通常カラー文字と白抜きノックアウト文字の両方を、同じグリフマスク基盤で実装できる
- 初期実装を無理に多機能化せず、横書き本文アートというコア要件に集中できる
