ComfyUI V3カスタムノード開発依頼書
1. 役割
あなたは、Python と ComfyUI の V3 カスタムノード開発に精通したエキスパート AI です。以下の仕様に基づき、既存の V1 カスタムノードを ComfyUI の V3 Node Schema に準拠した、完全な Python コードにリファクタリングしてください。

2. V3化するノードの概要
- **ノードID (クラス名)**: `ASCIIArtNodeV3`
- **表示名 (Display Name)**: `ASCII Art Generator V3`
- **カテゴリ (Category)**: `Image Processing/ASCII Art`
- **機能説明**: 入力画像をピクセル化・フォントレンダリングした ASCII アートに変換するノード。フォントサイズや文字セット、色補正、マスク合成などの高度なオプションを備え、バッチ処理にも対応します。

3. V3 Schema定義
- `io.Schema` の `inputs` と `outputs` は以下のように定義してください。
- 文字セットや色補正などのオプション入力は `optional=True` とし、既定値を `default` で供給します。

```python
from comfy_api.latest import io

class ASCIIArtNodeV3(io.ComfyNode):
    DOWNSCALE_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos", "contrast"]
    SHARPEN_MODES = ["None", "unsharp"]
    COLOR_MATCH_METHODS = ["mkl", "hm", "reinhard", "idt", "hm-mkl-hm"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        font_list = _load_fonts_with_fallback()
        dynamic_chars_default = (
            r"!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ "
        )

        return io.Schema(
            node_id="ASCIIArtNodeV3",
            display_name="ASCII Art Generator V3",
            category="Image Processing/ASCII Art",
            inputs=[
                io.Image.Input("image", display_name="Image"),
                io.Int.Input("pixel_size", default=20, min=1, max=200, step=1, display_name="Pixel Size"),
                io.Combo.Input("downscale_mode", options=cls.DOWNSCALE_MODES, default="area", display_name="Downscale Mode"),
                io.Float.Input("aspect_ratio_correction", default=0.75, min=0.1, max=10.0, step=0.05, display_name="Aspect Ratio Correction"),
                io.Combo.Input("font_name", options=font_list, display_name="Font"),
                io.Int.Input("font_size_min", default=8, min=1, max=100, step=1, display_name="Font Size Min"),
                io.Int.Input("font_size_max", default=16, min=1, max=200, step=1, display_name="Font Size Max"),
                io.Combo.Input("char_selection_mode", options=["brightness", "hue", "saturation", "luminance_hue"], default="luminance_hue", display_name="Character Selection"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, display_name="Seed"),
                io.Combo.Input("charset_source", options=["File", "Dynamic"], default="File", optional=True, display_name="Charset Source"),
                io.String.Input("ascii_chars_filename", default="set4.txt", optional=True, display_name="Charset File"),
                io.String.Input("dynamic_chars_to_test", default=dynamic_chars_default, multiline=True, optional=True, display_name="Dynamic Characters"),
                io.Float.Input("dynamic_sigma", default=1.5, min=0.1, max=10.0, step=0.1, optional=True, display_name="Dynamic Sigma"),
                io.Combo.Input("sharpen_mode", options=cls.SHARPEN_MODES, default="None", optional=True, display_name="Sharpen Mode"),
                io.Float.Input("sharpen_amount", default=1.0, min=0.0, max=5.0, step=0.1, optional=True, display_name="Sharpen Amount"),
                io.Float.Input("sharpen_threshold", default=0.0, min=0.0, max=1.0, step=0.01, optional=True, display_name="Sharpen Threshold"),
                io.Float.Input("brightness", default=1.0, min=0.0, max=5.0, step=0.05, optional=True, display_name="Brightness"),
                io.Float.Input("contrast", default=1.0, min=0.0, max=5.0, step=0.05, optional=True, display_name="Contrast"),
                io.Mask.Input("mask", optional=True, display_name="Mask"),
                io.Float.Input("mask_blend_radius", default=5.0, min=0.0, max=100.0, step=0.1, optional=True, display_name="Mask Blend Radius"),
                io.Combo.Input("mask_edge_adjustment", options=["None", "SmallerChars", "LowerDensity"], default="None", optional=True, display_name="Mask Edge Adjustment"),
                io.Float.Input("mask_edge_factor", default=3.0, min=0.0, max=20.0, step=0.1, optional=True, display_name="Mask Edge Factor"),
                io.Boolean.Input("enable_color_match", default=False, optional=True, display_name="Enable Color Match", label_on="Enabled", label_off="Disabled"),
                io.Combo.Input("color_match_method", options=cls.COLOR_MATCH_METHODS, default="mkl", optional=True, display_name="Color Match Method"),
                io.Combo.Input("log_level", options=["DEBUG", "INFO", "WARNING", "ERROR", "NONE"], default="INFO", optional=True, display_name="Log Level"),
            ],
            outputs=[
                io.Image.Output("image", display_name="ASCII Image"),
            ],
        )
```

4. コア処理ロジック (Core Logic)
- `execute` は入力を正規化した後、ログ設定・フォント解決・文字セットの読み込み／生成・マスク処理・ピクセル化・色補正・ASCII 描画・マスクブレンドを順に実行します。
- 失敗時は `torch.zeros_like(image)` を返し、成功時は `pil_to_tensor` でバッチを戻します。

```python
    @classmethod
    def execute(cls, ..., log_level: str = "INFO") -> io.NodeOutput:
        charset_source = charset_source or "File"
        ascii_chars_filename = ascii_chars_filename or "set4.txt"
        dynamic_chars_to_test = dynamic_chars_to_test or ""
        dynamic_sigma = 1.5 if dynamic_sigma is None else dynamic_sigma
        # 1. ログ設定と入力検証
        setup_logging(log_level, "ComfyUI.ASCIIArtNodeV3")
        # 2. フォントと文字セットの準備
        font_path = _resolve_font_path(font_name)
        ascii_sets = _load_or_generate_charset(font_path, ...)
        # 3. テンソル→PIL 変換とマスク処理
        pil_images = tensor_to_pil(image)
        mask_pil = _prepare_mask(mask, pil_images)
        # 4. 各画像を処理
        for idx, pil_image in enumerate(pil_images):
            pixelated = pixelate_image(...)
            if enable_color_match:
                pixelated = apply_color_match(...)
            ascii_image = create_ascii_art(...)
            final_image = apply_mask_blending(...) if mask_pil else ascii_image
            final_images.append(final_image)
        # 5. PIL→テンソルに戻して NodeOutput を返却
        output_tensor = pil_to_tensor(final_images)
        return io.NodeOutput(output_tensor)
```

5. 実装上の注意点
- V3 ノードは `io.ComfyNode` を継承し、`define_schema` と `execute` を実装してください。
- `NodeOutput` で出力 ID `image` に対応するテンソルを返却します。
- V1 用の `INPUT_TYPES` / `RETURN_TYPES` / `FUNCTION` / `CATEGORY` などは削除し、`NODE_CLASS_MAPPINGS` を更新してください。

```python
NODE_CLASS_MAPPINGS = {
    "ASCIIArtNodeV3": ASCIIArtNodeV3
}
```
