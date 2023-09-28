# mnn-segment-anything

## 模型文件
|   model   |  onnx  |   mnn  |
|:---------:|:------:|:------:|
| sam_vit_b | [![Download][download-b-onnx]][release-b-onnx] | [![Download][download-b-mnn]][release-b-mnn] |
| sam_vit_l | [![Download][download-l-onnx]][release-l-onnx] | [![Download][download-l-mnn]][release-l-mnn] |
| sam_vit_h | [![Download][download-h-onnx]][release-h-onnx] | [![Download][download-h-mnn]][release-h-mnn] |

[download-b-onnx]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_b_onnx/total
[download-b-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_b_mnn/total
[download-l-onnx]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_l_onnx/total
[download-l-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_l_mnn/total
[download-h-onnx]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_h_onnx/total
[download-h-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-segment-anything/vit_h_mnn/total
[release-b-onnx]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_b_onnx
[release-b-mnn]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_b_mnn
[release-l-onnx]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_l_onnx
[release-l-mnn]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_l_mnn
[release-h-onnx]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_h_onnx
[release-h-mnn]: https://github.com/wangzhaode/mnn-segment-anything/releases/tag/vit_h_mnn

- 端侧部署建议使用模型：
  - [embed_vitb_int4.mnn](https://github.com/wangzhaode/mnn-segment-anything/releases/download/vit_b_mnn/embed_vitb_int4.mnn): `54.5 MB`
  - [segment_vitb_fp32.mnn](https://github.com/wangzhaode/mnn-segment-anything/releases/download/vit_b_mnn/segment_vitb_fp32.mnn): `19.8M`

## 示例代码
- [Python](./python/)
- [C++](./cpp)

## 示例输出
![res](resource/res.jpg)
