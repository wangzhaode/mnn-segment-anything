![mnn-segment-anything](resource/logo.png)

# mnn-segment-anything

## 说明
`mnn-segment-anything`支持Python与C++在Linux, MacOS, Windows, Android与iOS上运行。目前支持以下模型：
- [segment-anything](https://github.com/facebookresearch/segment-anything)的`vit_b`, `vit_l`, `vit_h`
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)

## 模型文件

- 端侧部署建议使用模型：
  - edge_sam
    - [edge_embed.mnn](https://github.com/wangzhaode/mnn-segment-anything/releases/download/edge_mnn/edge_embed.mnn): `21 MB`
    - [edge_segment.mnn](https://github.com/wangzhaode/mnn-segment-anything/releases/download/edge_mnn/edge_segment.mnn): `19.3M`
  - mobiel_sam
    - [mobile_embed.mnn](https://github.com/wangzhaode/mnn-segment-anything/releases/download/mobile_mnn/mobile_embed.mnn): `26.7 MB`
    - [mobile_segment.mnn](https://github.com/wangzhaode/mnn-segment-anything/releases/download/mobile_mnn/mobile_segment.mnn): `19.7M`

|   model   |  onnx  |   mnn  |
|:---------:|:------:|:------:|
| edge_sam | [![Download][download-e-onnx]][release-e-onnx] | [![Download][download-e-mnn]][release-e-mnn] |
| mobile_sam | [![Download][download-m-onnx]][release-m-onnx] | [![Download][download-m-mnn]][release-m-mnn] |
| sam_vit_b | [![Download][download-b-onnx]][release-b-onnx] | [![Download][download-b-mnn]][release-b-mnn] |
| sam_vit_l | [![Download][download-l-onnx]][release-l-onnx] | [![Download][download-l-mnn]][release-l-mnn] |
| sam_vit_h | [![Download][download-h-onnx]][release-h-onnx] | [![Download][download-h-mnn]][release-h-mnn] |

[download-e-onnx]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/edge_onnx/total
[download-m-onnx]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/mobile_onnx/total
[download-b-onnx]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_b_onnx/total
[download-l-onnx]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_l_onnx/total
[download-h-onnx]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_h_onnx/total

[download-e-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/edge_mnn/total
[download-m-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/mobile_mnn/total
[download-b-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_b_mnn/total
[download-l-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_l_mnn/total
[download-h-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_h_mnn/total

[release-e-onnx]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/edge_onnx
[release-m-onnx]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/mobile_onnx
[release-b-onnx]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_b_onnx
[release-l-onnx]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_l_onnx
[release-h-onnx]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_h_onnx

[release-e-mnn]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/edge_mnn
[release-m-mnn]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/mobile_mnn
[release-b-mnn]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_b_mnn
[release-l-mnn]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_l_mnn
[release-h-mnn]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_h_mnn

## 示例代码
- [Python](./python/)
- [C++](./cpp)

## 示例输出
![res](resource/res.jpg)
