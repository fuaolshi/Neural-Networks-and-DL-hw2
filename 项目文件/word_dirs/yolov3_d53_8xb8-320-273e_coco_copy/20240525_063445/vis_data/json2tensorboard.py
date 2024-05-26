import json
from torch.utils.tensorboard import SummaryWriter

# Load the JSON file
with open(r'/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/3/20240525_063445/vis_data/scalars.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Create a SummaryWriter to write the logs
writer = SummaryWriter(log_dir=r"/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/3/20240525_063445/vis_data/tensorboard/")

# Loop through the data and add scalars to TensorBoard
for entry in data:
    step = entry.pop('step')
    for key, value in entry.items():
        writer.add_scalar(key, value, step)

# Close the writer
writer.flush()
writer.close()
print("写入成功")