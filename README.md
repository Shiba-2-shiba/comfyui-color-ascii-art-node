# ComfyUI-color-ascii-art-node


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
cd Yourdirectory/ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/comfyui-color-ascii-art-node.git
cd comfyui-color-ascii-art-node
pip install -r requirements.txt

```

## Usage

![Example Workflow](https://github.com/Shiba-2-shiba/comfyui-color-ascii-art-node/blob/main/ref_image/flow_exmaple.png)

「ASCIIARTNode」「ASCIIARTNodev2」「ASCIIArtSinglefontNode」がノードに追加されます。

これらのノードは、「image」の入力のみに対応し、「Image」の出力を行います。

パラメーターを調整することで、出力画像を変えることが出来ます。


ASCIIARTNode”, ‘ASCIIARTNodev2’, and ‘ASCIIARTSinglefontNode’ will be added to the nodes.

This node accepts only “image” input and outputs “Image”.

The output image can be changed by adjusting the parameters.


現在使用出来るノードの種類の比較です。

「ASCIIARTNode」：3種類のサイズのフォントが配置されますが、重複があります。アスペクト比の増減で改善されます。

「ASCIIARTNodev2」：重複を基本しないように設定しています。フォントのサイズはランダムです。

「ASCIIArtSinglefontNode」：重複を基本しないように設定しています。フォントのサイズが単一です。


Here is a comparison of the types of nodes currently available

ASCIIARTNode": Three font sizes are placed, but there is some overlap. This can be improved by increasing or decreasing the aspect ratio.

ASCIIARTNodev2": The duplication is not basic. The font size is random.

ASCIIArtSinglefontNode": This is set so that duplication is not basic. The font size is single.


![Example Workflow](https://github.com/Shiba-2-shiba/comfyui-color-ascii-art-node/blob/main/ref_image/flowexmaple_3.png)


## Parameters


![Example Workflow](https://github.com/Shiba-2-shiba/comfyui-color-ascii-art-node/blob/main/ref_image/Asciiartnode.png)


①pixel size：ピクセル化するサイズの数値です。デフォルトは8にしています。

②font_size_min：ここで設定したサイズと、２倍、３倍のサイズのフォントが使用されます(ASCIIARTNodeのみ)。

③aspect_ratio_correction：文字の重複が目立つ際にここの数値を大きくして調整します。

④font_name：fontフォルダ内にあるフォントファイルのリストから選択できます。デフォルトはChewy-Regular.ttfです。好きなフォントファイルを追加で配置することで別なフォントが使用できます。

⑤ascii_chars_filename：ディレクトリ内のset1.txtと入力してください。この中の文字を変えることで、配置される文字を変えることができます。set2-set5がありますが、文字に対応していないフォントは白紙になります。

⑥brightness：画像の明るさの調整をします。基本は1.0です。色が薄いところを白くしたい場合はこの数値を上げます。

⑦contrast：画像のコントラストを上げます。デフォルトは1.0ですが、このみによって上げてください。


＜English＞

①pixel size: The numerical value of the size to be pixelated. Default is set to 8.

②font_size_min: The size of the font will be twice or three times the size set here（ASCIIARTNode）.

③aspect_ratio_correction：When there is a noticeable overlap of characters, increase the value here to adjust it.

④font_name：Enter the file name of the font file in the directory. The default is Chewy-Regular.ttf. You can use a different font by placing a font file of your choice.

⑤ascii_chars_filename: Enter ascii_custum_characters.txt in the directory. By changing the characters in this file, you can change the characters to be placed.

⑥brightness: Adjust the brightness of the image. The basic value is 1.0. If you want to make light-colored areas whiter, increase this value.

⑦contrast: Increases the contrast of the image. The default value is 1.0, but you can increase this value depending on the contrast.








