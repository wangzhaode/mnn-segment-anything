# Usage

## Install MNN
```
pip install MNN
```

## Run Demo
```
python segment_anything_example.py --embed embed.mnn --sam segment.mnn --img ../resource/truck.jpg
# edge model need add `--edge`
python segment_anything_example.py --embed edge_embed.mnn --sam edge_segment.mnn --img ../resource/truck.jpg --edge
```