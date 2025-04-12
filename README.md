# ComfyUI-color-ascii-art-node


png画像をカラーのアスキーアート化するカスタムノードです。

以下に記載がありますが、複数のフォントサイズが使用される仕様になっています。

生成されるファイルの解像度は入力画像と同じになるように設定しています。

This is a custom node to convert png images into color ASCII art.

As noted below, multiple font sizes are used in the specification.

The resolution of the generated file is set to be the same as the input image.


## 修正歴

2025/4/12　カスタムノードをv3のみにし、大幅に内容を修正しました。

2024/9/26　カスタムノードにシード値を追加しました。

2024/9/18　カスタムノードのデフォルト値を修正しました。

2024/9/5　カスタムノードにmaskの入力を追加しました。若干処理速度を上げるスクリプトを追加しました。

2024/9/3　カスタムノードの種類を3種類に追加しました。


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

![Example Workflow](https://github.com/Shiba-2-shiba/comfyui-color-ascii-art-node/blob/main/ref_image/img1.png)

「ASCIIARTNode v3」がノードに追加されます。

このノードは、「image」と「mask」の入力に対応し、カラーアスキー化した「Image」の出力を行います。

パラメーターを調整することで、出力画像を変えることが出来ます。


ASCIIARTNode v3” will be added to the node.

This node responds to “image” and “mask” inputs and outputs a color ASCII-ized “Image”.

The output image can be changed by adjusting the parameters.



## Parameters


![Example Workflow](https://github.com/Shiba-2-shiba/comfyui-color-ascii-art-node/blob/main/ref_image/img2.png)


①pixel size：ピクセル化するサイズの数値です。デフォルトは20にしています。ピクセルサイズとフォントサイズが同じだと綺麗に配置される傾向があります。

②font_size_min：ここで設定したサイズと、２倍、３倍のサイズのフォントが使用されます(ASCIIARTNodeのみ)。

③aspect_ratio_correction：文字の重複が目立つ際にここの数値を大きくして調整します。

④font_name：fontフォルダ内にあるフォントファイルのリストから選択できます。デフォルトはChewy-Regular.ttfです。

⑤ascii_chars_filename：ディレクトリ内のset1.txtと入力してください。この中の文字を変えることで、配置される文字を変えることができます。set1-set5がありますが、文字に対応していないフォントを選択した場合は出力が白紙になります。

⑥brightness：画像の明るさの調整をします。基本は1.0です。色が薄いところを白くしたい場合はこの数値を上げます。

⑦contrast：画像のコントラストを上げます。デフォルトは1.0ですが、このみによって上げてください。

⑧seed：文字の選択のランダム化を調整しています。









