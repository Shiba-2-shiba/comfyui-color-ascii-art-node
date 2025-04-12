# colormatch.py (Swapped Src/Ref for ColorMatcher + Debugging)
import logging
import numpy as np
from PIL import Image
import traceback

# --- Logger Setup ---
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3.ColorMatch")
# ... (logger setup) ...

# --- Color Matcher Import ---
_color_matcher_available = False
try:
    import color_matcher
    _color_matcher_available = True
except ImportError:
    _color_matcher_available = False
    logger.warning("color_matcherライブラリが読み込めませんでした。")
except Exception as import_e:
    _color_matcher_available = False
    logger.error("color_matcher読み込み中にエラーが発生しました", exc_info=True)

def apply_color_match(
    source_image_pil: Image.Image, # 元の画像 (1024x1024)
    target_image_pil: Image.Image, # ピクセル化後の画像 (64x85)
    method: str = 'mkl'
    ) -> Image.Image:
    """
    ピクセル化後の画像(target)に、元の画像(source)の色を転送します。
    ColorMatcherには src=target, ref=source として渡します。
    """
    logger.debug("Entering apply_color_match function.")

    # --- Check availability ---
    if not _color_matcher_available:
        logger.warning("Color matching skipped because color-matcher library is not available.")
        # 元のピクセル化画像をそのまま返す
        return target_image_pil

    # --- Input Validation (PIL Images) ---
    if source_image_pil is None:
        logger.error("Source image (original) for color matching is None. Returning target image.")
        return target_image_pil
    if target_image_pil is None:
        logger.error("Target image (pixelated) for color matching is None. Cannot perform color matching.")
        return target_image_pil # Noneが返る

    logger.info(f"Applying color matching using method: '{method}' (Source=Original, Target=Pixelated -> Swapping for Lib: Src=Pixelated, Ref=Original)")
    logger.debug(f"Original Source image size: {source_image_pil.size}, mode: {source_image_pil.mode}")
    logger.debug(f"Original Target image size (Pixelated): {target_image_pil.size}, mode: {target_image_pil.mode}")

    try:
        # --- 画像をNumPy配列に変換 ---
        # source_np は元の画像 (リファレンスとして使用)
        logger.debug("Converting original PIL image (Ref) to NumPy array (RGB, uint8)...")
        ref_np = np.array(source_image_pil.convert('RGB'), dtype=np.uint8)
        logger.debug(f"Ref conversion complete. Shape: {ref_np.shape}, Dtype: {ref_np.dtype}")

        # target_np はピクセル化画像 (ソースとして使用)
        logger.debug("Converting pixelated PIL image (Src) to NumPy array (RGB, uint8)...")
        src_np = np.array(target_image_pil.convert('RGB'), dtype=np.uint8)
        logger.debug(f"Src conversion complete. Shape: {src_np.shape}, Dtype: {src_np.dtype}")

        # --- NumPy配列のチェック ---
        # (チェック対象を src_np と ref_np に変更)
        logger.debug("Performing NumPy array checks...")
        valid_input = True
        # ... (src_np と ref_np に対する None, ndim, shape[2], size チェック) ...
        if not valid_input:
            logger.error("NumPy array validation failed. Returning original pixelated image.")
            return target_image_pil
        logger.debug("NumPy array checks passed.")

        # --- color-matcherによる色転送の実行 ---
        logger.debug("Initializing color_matcher.ColorMatcher (Src=Pixelated, Ref=Original)...")

        # --- ★★★ 初期化方法: src と ref を入れ替えて渡す ★★★ ---
        cm = color_matcher.ColorMatcher(src=src_np, ref=ref_np)
        logger.warning("Swapped Source(Original) and Target(Pixelated) for ColorMatcher library call: src=pixelated_np, ref=original_np")

        logger.debug(f"ColorMatcher object initialized: {cm}")

        # --- デバッグチェック (維持) ---
        initialization_ok = True
        if not hasattr(cm, '_src') or cm._src is None:
            logger.error("FATAL: cm._src (Pixelated) is None or missing after initialization!")
            initialization_ok = False
        else:
             logger.debug(f"cm._src (Pixelated) is OK after init. Shape: {cm._src.shape}, Dtype: {cm._src.dtype}")

        if not hasattr(cm, '_ref') or cm._ref is None:
             logger.warning("cm._ref (Original) is None or missing after initialization.")
        else:
             logger.debug(f"cm._ref (Original) is OK after init. Shape: {cm._ref.shape}, Dtype: {cm._ref.dtype}")

        if not initialization_ok:
             logger.error("ColorMatcher initialization failed. Returning original pixelated image.")
             return target_image_pil
        # --- デバッグチェック完了 ---

        logger.debug(f"Calling cm.transfer(method='{method}')...")
        # transfer は src (ピクセル化画像) のサイズで結果を返すはず
        result_np = cm.transfer(method=method)
        logger.debug("cm.transfer() method completed.")

        # --- 結果のチェックと変換 ---
        if result_np is None:
            logger.error("ColorMatcher.transfer() returned None. Returning original pixelated image.")
            return target_image_pil

        if not isinstance(result_np, np.ndarray):
             logger.error(f"ColorMatcher.transfer() returned unexpected type: {type(result_np)}. Returning original pixelated image.")
             return target_image_pil
        logger.debug(f"Result NumPy array - Shape: {result_np.shape}, Dtype: {result_np.dtype}") # ここが (85, 64, 3) になるはず

        # uint8 への変換
        if result_np.dtype != np.uint8:
             logger.warning(f"Result NumPy array dtype is {result_np.dtype}, converting to uint8 for PIL conversion.")
             result_np = np.clip(result_np, 0, 255).astype(np.uint8)

        # PIL への変換前の形状チェック
        if result_np.ndim != 3 or result_np.shape[2] != 3:
             logger.error(f"Result NumPy array has invalid shape {result_np.shape} for RGB PIL conversion. Returning original pixelated image.")
             return target_image_pil

        # --- 結果をPIL画像に変換 ---
        logger.debug("Converting result NumPy array back to PIL image...")
        matched_image_pil = Image.fromarray(result_np, mode='RGB')
        logger.debug(f"Conversion back to PIL complete. Size: {matched_image_pil.size}") # サイズもログに出力

        logger.info("Color matching process finished successfully (applied original colors to pixelated image).")
        # この matched_image_pil がカラーマッチされたピクセル化画像となる
        return matched_image_pil

    except Exception as e:
        # 予期せぬエラーをキャッチ
        logger.error(f"Error during color matching process in apply_color_match: {e}", exc_info=True)
        logger.warning("Color matching failed due to an error, returning original pixelated image.")
        return target_image_pil # エラー時は元のピクセル化画像を返す

# --- Example Usage (for testing standalone) ---
# ...