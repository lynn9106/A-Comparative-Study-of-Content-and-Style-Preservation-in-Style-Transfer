import torch
import torchvision.models as models

# vgg19 = models.vgg19(pretrained=True)

# torch.save(vgg19.state_dict(), './weights/vgg19-dcbb9e9d.pth')

print(torch.cuda.is_available())  # 應該返回 True
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.current_device())  # 顯示當前設備ID
# print(torch.cuda.get_device_name(0))  # 顯示顯卡名稱