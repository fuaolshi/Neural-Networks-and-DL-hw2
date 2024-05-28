import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
import os
import matplotlib.pyplot as plt
from mmcv.visualization.image import imshow_det_bboxes

# 配置文件路径
faster_rcnn_config_file = '/mnt/ly/models/mmdetection/mmdetection-main/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
yolov3_config_file = '/mnt/ly/models/mmdetection/mmdetection-main/configs/yolo/yolov3_d53_8xb8-320-273e_coco_copy.py'

# 训练好的模型权重文件路径
faster_rcnn_checkpoint_file = '/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/faster-rcnn/1/epoch_8.pth'
yolov3_checkpoint_file = '/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/3/epoch_273.pth'

# 初始化检测模型
faster_rcnn_model = init_detector(faster_rcnn_config_file, faster_rcnn_checkpoint_file, device='cuda:0')
yolov3_model = init_detector(yolov3_config_file, yolov3_checkpoint_file, device='cuda:0')

# VOC数据集类别
class_names = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 图片文件夹路径
img_folder = "/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/test_1/out_picture/picture"
# 结果保存路径
output_folder = "/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/test_1/out_picture/result/Comparison"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有图片文件
# img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
img_files = ['car bus person.jpg']

# 对每张图片进行推理并保存对比结果
for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)

    # Faster R-CNN 推理
    faster_rcnn_result = inference_detector(faster_rcnn_model, img_path)
    faster_rcnn_bboxes = faster_rcnn_result.pred_instances.bboxes.cpu().numpy()
    faster_rcnn_labels = faster_rcnn_result.pred_instances.labels.cpu().numpy()
    faster_rcnn_scores = faster_rcnn_result.pred_instances.scores.cpu().numpy()
    faster_rcnn_bboxes_with_scores = np.hstack([faster_rcnn_bboxes, faster_rcnn_scores[:, np.newaxis]])

    # YOLOv3 推理
    yolov3_result = inference_detector(yolov3_model, img_path)
    yolov3_bboxes = yolov3_result.pred_instances.bboxes.cpu().numpy()
    yolov3_labels = yolov3_result.pred_instances.labels.cpu().numpy()
    yolov3_scores = yolov3_result.pred_instances.scores.cpu().numpy()
    yolov3_bboxes_with_scores = np.hstack([yolov3_bboxes, yolov3_scores[:, np.newaxis]])

    # 读取图像
    img = mmcv.imread(img_path)

    # 可视化并保存结果
    faster_rcnn_img = imshow_det_bboxes(
        img.copy(),
        faster_rcnn_bboxes_with_scores,
        faster_rcnn_labels,
        class_names=class_names,
        score_thr=0.3,
        show=False
    )

    yolov3_img = imshow_det_bboxes(
        img.copy(),
        yolov3_bboxes_with_scores,
        yolov3_labels,
        class_names=class_names,
        score_thr=0.3,
        show=False
    )

    # 创建对比图
    fig, axs = plt.subplots(1, 2, figsize=(15, 10), dpi=300)
    axs[0].imshow(faster_rcnn_img)
    axs[0].set_title('Faster R-CNN')
    axs[0].axis('off')
    axs[1].imshow(yolov3_img)
    axs[1].set_title('YOLOv3')
    axs[1].axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    # 保存对比图
    comparison_output_path = os.path.join(output_folder, img_file)
    plt.savefig(comparison_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
