# charset_generator.py
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage
from typing import Dict

# --- Logger Setup ---
# 他のモジュールと同様のロガー設定
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3.CharsetGenerator")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)


def _calculate_char_density(char: str, font: ImageFont.FreeTypeFont, sigma: float) -> float:
    """
    単一の文字を描画し、ガウシアンブラーを適用してその「密度」を計算する。
    密度は、ぼかした後の画像の平均輝度として定義される。

    Args:
        char (str): 密度を計算する文字。
        font (ImageFont.FreeTypeFont): 描画に使用するフォントオブジェクト。
        sigma (float): ガウシアンブラーの強さ。値が大きいほどぼかしが強くなる。

    Returns:
        float: 計算された文字の密度 (0.0 - 255.0 の範囲)。
    """
    try:
        # 文字をピッタリ囲むバウンディングボックスを取得
        # getbboxは (left, top, right, bottom) を返す
        bbox = font.getbbox(char)
    except OSError:
        # 一部のフォント/文字で発生する可能性のあるOSレベルのエラー
        logger.warning(f"Could not get bbox for character '{char}'. Skipping.")
        return 0.0

    if not bbox or bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
        # スペースなど、描画領域を持たない文字は密度0とする
        return 0.0

    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    # 余白を持たせた画像を作成 (ブラーが端で切れないようにするため)
    padding = int(sigma * 2)
    img_width = char_width + padding * 2
    img_height = char_height + padding * 2
    
    # 黒背景(0)のグレースケール画像('L')を作成
    image = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(image)

    # 画像の中央に白文字(255)で描画
    # 描画位置は、バウンディングボックスの左上オフセットとパディングを考慮
    draw_x = padding - bbox[0]
    draw_y = padding - bbox[1]
    draw.text((draw_x, draw_y), char, font=font, fill=255)

    # PIL ImageをNumPy配列に変換
    # これにより、ピクセルデータが2D配列として扱える
    numpy_image = np.array(image, dtype=np.float32)

    # NumPy配列にガウシアンブラーを適用
    # これが「畳み込み」処理に相当し、文字の「詰まり具合」を評価する核となる
    blurred_image = scipy.ndimage.gaussian_filter(numpy_image, sigma=sigma)

    # ぼかした後の画像の平均値を密度として返す
    # 詰まった文字ほど白(255)が残り、平均値は高くなる
    density = blurred_image.mean()
    
    return float(density)


def generate_dynamic_charset(
    font_path: str,
    font_size: int = 32,
    characters_to_test: str = R"""!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ """,
    sigma: float = 1.5
) -> str:
    """
    指定されたフォントから、文字の視覚的な「密度」に基づいた
    ASCII文字セットを動的に生成する。

    Args:
        font_path (str): .ttfまたは.otfフォントファイルへのパス。
        font_size (int): 密度の計算に使用するフォントサイズ。大きい方が精度が上がるが遅くなる。
        characters_to_test (str): 密度を測定し、セットに含める候補となる文字の文字列。
        sigma (float): 文字の密度計算に使うガウシアンブラーの強さ。
                         フォントの太さに応じて調整が必要 (1.0 - 2.5程度が一般的)。

    Returns:
        str: 密度の低い文字から高い文字へとソートされた文字列。
             エラーが発生した場合は空文字列を返す。
    """
    logger.info(f"Starting dynamic charset generation for font: {font_path}")
    logger.debug(f"Parameters: font_size={font_size}, sigma={sigma}, {len(characters_to_test)} chars to test.")

    try:
        # フォントを読み込む
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logger.error(f"Failed to load font from {font_path}: {e}", exc_info=True)
        return ""

    # 各文字とその密度を格納する辞書
    densities: Dict[str, float] = {}

    for char in characters_to_test:
        if char not in densities: # 重複文字をスキップ
            densities[char] = _calculate_char_density(char, font, sigma)

    # 辞書を密度(値)に基づいてソートする
    # sorted()はキーのリストを返す
    try:
        sorted_chars = sorted(densities, key=lambda k: densities[k])
    except Exception as e:
        logger.error(f"Failed to sort characters by density: {e}", exc_info=True)
        return ""

    # ソートされた文字のリストを一つの文字列に結合
    dynamic_charset = "".join(sorted_chars)

    logger.info(f"Successfully generated dynamic charset with {len(dynamic_charset)} characters.")
    logger.debug(f"Generated set (excerpt): {dynamic_charset[:20]}...{dynamic_charset[-20:]}")
    
    return dynamic_charset

# --- このファイルを直接実行した際のテスト用コード ---
if __name__ == '__main__':
    # このテストを実行するには、'Arial.ttf'などのフォントファイルが必要です
    # パスは適宜変更してください
    import os
    font_name = 'Arial.ttf'
    # Windows/Mac/Linuxでの一般的なフォントパスを探す
    font_path = ''
    if os.name == 'nt': # Windows
        font_path = os.path.join(os.environ['WINDIR'], 'Fonts', font_name)
    elif os.name == 'posix': # macOS or Linux
        for path in ['/System/Library/Fonts/Supplemental/', '/usr/share/fonts/truetype/msttcorefonts/']:
            if os.path.exists(os.path.join(path, font_name)):
                font_path = os.path.join(path, font_name)
                break
    
    if os.path.exists(font_path):
        print(f"Testing with font: {font_path}")
        # 動的に文字セットを生成
        generated_set = generate_dynamic_charset(font_path=font_path, font_size=48, sigma=2.0)
        
        print("\n--- Generated Dynamic Charset ---")
        print(generated_set)
        print(f"\nTotal characters: {len(generated_set)}")
    else:
        print(f"Test font '{font_name}' not found. Please specify a valid font path.")