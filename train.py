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
# results = model.train(data="ultralytics/datasets/rain.yaml", epochs=20, batch=-1)  # 训练模型
if __name__ == '__main__':
    # Use the model
    results = model.train(data="ultralytics\data\dengjianji.yaml", imgsz=640, epochs=20, batch=16, workers=10)  # 训练模型
    # results = model.val()  
    # results = model("自己的验证图片")  
    success = YOLO("yolov8n.pt").export(format="onnx") 