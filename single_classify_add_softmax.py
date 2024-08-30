import torch
import torch.nn as nn
import torch.nn.functional as F

in_model_path = r"D:\Zhiyuan\video\record_all_finall\classify\crop\model\v2\vest_enb0_C3_E20_0.00483_0.99906_0.01215_0.99811_noNM.pt"
out_model_path = r"D:\Zhiyuan\video\record_all_finall\classify\crop\model\v2\vest_enb0_C3_E20_0.00483_0.99906_0.01215_0.99811_noNM_softmax.pt"
# 加载.pt文件中的模型
model = torch.load(in_model_path)
model.eval()  # 将模型设置为评估模式

# 定义一个新的模型，它在原始模型的输出上增加了一个Softmax层
class CustomModel(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomModel, self).__init__()
        self.pretrained_model = pretrained_model

    def forward(self, x):
        outputs = self.pretrained_model(x)
        probabilities = F.softmax(outputs, dim=1)
        return probabilities

# 实例化新模型
custom_model = CustomModel(model)

# 假设custom_model是您更新后的模型实例
# 保存模型到新的.pt文件
torch.save(custom_model, out_model_path)
