import torch 
import torch.nn.functional as F

class CIFARCNN(torch.nn.Module): 
    def __init__(self, input_channels: int ):
        super().__init__()
        # input image is 32x32 (HxW) x 3 (C) x N
        self.conv_1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2) # spatial dimensions reduce to 16x16 
        self.conv_3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2) # spatial dimensions reduce to 8x8
        self.flatten = torch.nn.Flatten()
        self.fc_1 = torch.nn.Linear(64*8*8, 512)
        self.fc_2 = torch.nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.maxpool_1(x)

        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = self.maxpool_2(x)

        x = self.flatten(x)
        return self.fc_2(self.fc_1(x))

# test       
# if __name__ == "__main__": 
#     model = CIFARCNN(input_channels=3)
#     test_input = torch.rand([1, 3, 32, 32]) # N C H W - default pytorch format

#     model.eval()
#     out = model(test_input)
#     print(out.shape)