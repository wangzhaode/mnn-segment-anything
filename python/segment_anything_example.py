#-- coding:utf8 --
import argparse
import time
import MNN
import MNN.numpy as np
import MNN.cv as cv2

def inference(emed, sam, img, precision, backend, thread):
    mask_threshold = 0.0
    # 0. load model
    config = {}
    config['precision'] = precision
    config['backend'] = backend
    config['numThread'] = thread
    rt = MNN.nn.create_runtime_manager((config,))
    embed = MNN.nn.load_module_from_file(emed, [], [], runtime_manager=rt)
    sam = MNN.nn.load_module_from_file(sam,
         ['point_coords', 'point_labels', 'image_embeddings', 'has_mask_input', 'mask_input', 'orig_im_size'],
         ['iou_predictions', 'low_res_masks', 'masks'], runtime_manager=rt)
    # 1. preprocess
    image = cv2.imread(img)
    origin_h, origin_w, _ = image.shape
    length = 1024
    if origin_h > origin_w:
        new_w = round(origin_w * float(length) / origin_h)
        new_h = length
    else:
        new_h = round(origin_h * float(length) / origin_w)
        new_w = length
    scale_w = new_w / origin_w
    sclae_h = new_h / origin_h
    input_var = cv2.resize(image, (new_w, new_h), 0., 0., cv2.INTER_LINEAR, -1, [123.675, 116.28, 103.53], [1/58.395, 1/57.12, 1/57.375])
    input_var = np.pad(input_var, [[0, length - new_h], [0, length - new_w], [0, 0]], 'constant')
    input_var = np.expand_dims(input_var, 0)
    # 2. embedding forward
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    t1 = time.time()
    output_var = embed.forward(input_var)
    t2 = time.time()
    print('# 1. embedding times: {} ms'.format((t2 - t1) * 1000))
    image_embedding = MNN.expr.convert(output_var, MNN.expr.NCHW)
    # 3. segment forward
    points = [[500, 375]]
    sclaes = [scale_w, sclae_h]
    input_point = np.array(points) * sclaes
    input_label = np.array([1])
    point_coords = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    point_labels = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    orig_im_size = np.array([float(origin_h), float(origin_w)], dtype=np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.zeros(1, dtype=np.float32)
    t1 = time.time()
    output_vars = sam.onForward([point_coords, point_labels, image_embedding, has_mask_input, mask_input, orig_im_size])
    t2 = time.time()
    print('# 1. segment times: {} ms'.format((t2 - t1) * 1000))
    masks = MNN.expr.convert(output_vars[2], MNN.expr.NCHW)
    masks = masks.squeeze([0])[0]
    # 4. postprocess: draw masks and point
    masks = (masks > mask_threshold).reshape([origin_h, origin_w, 1])
    color = np.array([30, 144, 255]).reshape([1, 1, -1])
    image = (image + masks * color).astype(np.uint8)
    for point in points:
        cv2.circle(image, point, 10, (0, 0, 255), 5)
    cv2.imwrite('res.jpg', image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', type=str, required=True, help='the embedding model path')
    parser.add_argument('--sam', type=str, required=True, help='the sam model path')
    parser.add_argument('--img', type=str, required=True, help='the input image path')
    parser.add_argument('--precision', type=str, default='normal', help='inference precision: normal, low, high, lowBF')
    parser.add_argument('--backend', type=str, default='CPU', help='inference backend: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI')
    parser.add_argument('--thread', type=int, default=4, help='inference using thread: int')
    args = parser.parse_args()
    inference(args.embed, args.sam, args.img, args.precision, args.backend, args.thread)