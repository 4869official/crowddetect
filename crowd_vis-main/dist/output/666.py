import paddle
import cv2
import numpy as np


# 定义模型结构，这里需要根据实际模型结构来定义
class PP_TSM(paddle.nn.Layer):
    def __init__(self):
        super(PP_TSM, self).__init__()
        # 以下仅为示例结构，你需要根据你的模型.pdmodel文件来构建模型
        # 示例：添加一个简单的卷积层
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 假设还有其他层，如池化层、全连接层等
        # self.pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        # self.fc = paddle.nn.Linear(in_features=64*56*56, out_features=10)

    def forward(self, x):
        # 定义你的前向传播过程
        x = self.conv1(x)
        # x = self.pool(x)
        # x = paddle.flatten(x, 1)
        # x = self.fc(x)
        return x


# 创建模型实例
model = PP_TSM()

# 加载模型参数
params_path = 'C:\\Users\\19025\\Desktop\\crowd_vis-main\\pp-human\\pipeline\\model\\ppTSM\\ppTSM_fight.pdparams'
model_state_dict = paddle.load(params_path)
model.set_state_dict(model_state_dict)

# 将模型设置为评估模式
model.eval()


# 读取图片并预处理
def preprocess(image_path):
    # 读取图片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # 假设模型输入为224x224
    img = img.transpose((2, 0, 1)).astype('float32')
    img /= 255.0  # 归一化
    return img


# 图片路径
image_path = 'C:\\Users\\19025\\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian\\20241031_150749.jpg'
img = preprocess(image_path)

# 增加批次维度
img = paddle.to_tensor(img[np.newaxis, ...])

# 执行推理
output = model(img)

# 输出结果
print(output)