# #%%
# from ultralytics import YOLO
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


# model = YOLO("yolov8n.pt")
 

# if __name__ == '__main__':

#     # 在这里添加你的主要代码,比如:
#     results = model.train(data="ultralytics\data\dengjianji.yaml",imgsz=640, epochs=20, batch=16, workers=0)

#%%
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 加载模型
model = YOLO("yolov8n.pt")  # 从头开始构建新模型
# model = YOLO("weights/yolov8n.pt")  # 加载预训练模型（推荐用于训练）
 
# Use the model
# results = model.train(data="ultralytics/datasets/rain.yaml", epochs=10, batch=-1)  # 训练模型
if __name__ == '__main__':
    # Use the model
    results = model.train(data="ultralytics\data\dengjianji.yaml", imgsz=640, epochs=10, batch=16, workers=10)  # 训练模型
    # results = model.val()  
    # results = model("自己的验证图片")  
    success = YOLO("yolov8n.pt").export(format="onnx") 

#%%  训练
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load a model
model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model 不使用预训练权重，就注释这一行即可
if __name__ == '__main__':
    # train
    model.train(data='ultralytics\data\dengjianji.yaml',
                    cache=False,
                    imgsz=640,
                    epochs=20,
                    batch=32,
                    close_mosaic=0,
                    workers=4,
                    device='0',
                    optimizer='SGD', # using SGD
                    amp=False, # close amp
                    project='runs/train',
                    name='exp',
                    )


# %%  验证值
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='ultralytics\data\dengjianji.yaml',
              split='val',
              imgsz=640,
              batch=16,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )

# %%  测试值
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='ultralytics\data\dengjianji.yaml',
              split='test',
              imgsz=640,
              batch=16,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )