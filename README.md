# comfui-color-ascii-art-node
png画像をカラーのアスキーアート化するカスタムノードです。
以下に記載がありますが、複数のフォントサイズが使用される仕様になっています。
生成されるファイルの解像度は入力画像と同じになるように設定しています。

This is a custom node to convert png images into color ASCII art.
As noted below, multiple font sizes are used in the specification.
The resolution of the generated file is set to be the same as the input image.

## Installation
ComfyUIのカスタムノードディレクトリにインストールします。

Clone the repository and navigate to the project directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/comfyui-color-ascii-art-node.git
cd comfyui-color-ascii-art-node
pip install -r requirements.txt

```

## Usage
「ASCIIARTNode」がノードに追加されます。
このノードは、「image」の入力のみに対応し、「Image」の出力を行います。
パラメーターを調整することで、出力画像を変えることが出来ます。

"ASCIIARTNode” will be added to the node.
This node accepts only “image” input and outputs “Image”.
The output image can be changed by adjusting the parameters.


## Parameters


①pixel size：ピクセル化するサイズの数値です。デフォルトは8にしています。

②font_size_min：ここで設定したサイズと、２倍、３倍のサイズのフォントが使用されます。

③aspect_ratio_correction：文字の重複が目立つ際にここの数値を大きくして調整します。

④font_name：ディレクトリ内のフォントファイルのファイル名を入力してください。デフォルトはChewy-Regular.ttfです。好きなフォントファイルを配置することで別なフォントが使用できます。

⑤ascii_chars_filename：ディレクトリ内のascii_custum_characters.txtと入力してください。この中の文字を変えることで、配置される文字を変えることができます。

⑥brightness：画像の明るさの調整をします。基本は1.0です。色が薄いところを白くしたい場合はこの数値を上げます。

⑦contrast：画像のコントラストを上げます。デフォルトは1.0ですが、このみによって上げてください。

image: The input image for generating ASCII art.
pixel_size: Size of the pixels in the ASCII art.
font_size_min: Minimum font size used for ASCII characters.
aspect_ratio_correction: Adjustment for aspect ratio.
font_name: The font file to be used.
ascii_chars_filename: The text file containing custom ASCII character sets.
brightness: Adjust the brightness of the input image.
contrast: Adjust the contrast of the input image.
Example
Below is an example of the generated ASCII art:






