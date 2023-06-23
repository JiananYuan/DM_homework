Input
  |
Conv1d (in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3)
  |
BatchNorm1d
  |
ReLU
  |
MaxPool1d (kernel_size=2)
  |
Conv1d (in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
  |
BatchNorm1d
  |
ReLU
  |
MaxPool1d (kernel_size=2)
  |
Conv1d (in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
  |
BatchNorm1d
  |
ReLU
  |
Conv1d (in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
  |
BatchNorm1d
  |
ReLU
  |
MaxPool1d (kernel_size=2)
  |
Linear (in_features=6400, out_features=4096)
  |
BatchNorm1d
  |
ReLU
  |
Linear (in_features=4096, out_features=64)
  |
BatchNorm1d
  |
ReLU
  |
Linear (in_features=64, out_features=4)
  |
Output



Input
  |
Conv1d (in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
  |
BatchNorm1d
  |
ReLU
  |
Conv1d (in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5)
  |
BatchNorm1d
  |
ReLU
  |
MaxPool1d (kernel_size=4)
  |
Conv1d (in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
  |
BatchNorm1d
  |
ReLU
  |
Conv1d (in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
  |
BatchNorm1d
  |
ReLU
  |
MaxPool1d (kernel_size=4)
  |
Dropout (p=0.1)
  |
Flatten
  |
Linear (in_features=3072, out_features=1024)
  |
ReLU
  |
Linear (in_features=1024, out_features=128)
  |
ReLU
  |
Linear (in_features=128, out_features=4)
  |
Softmax
  |
Output



Input
  |
Conv1d (in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
  |
BatchNorm1d
  |
ReLU
  |
Conv1d (in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5)
  |
ReLU
  |
MaxPool1d (kernel_size=4)
  |
Conv1d (in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
  |
ReLU
  |
Conv1d (in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
  |
ReLU
  |
MaxPool1d (kernel_size=4)
  |
Dropout (p=0.1)
  |
Flatten
  |
Linear (in_features=3072, out_features=1024)
  |
ReLU
  |
Linear (in_features=1024, out_features=128)
  |
ReLU
  |
Linear (in_features=128, out_features=4)
  |
Softmax
Output


Input
  |
Conv1d (in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
  |
BatchNorm1d
  |
LeakyReLU
  |
Conv1d (in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5)
  |
BatchNorm1d
  |
LeakyReLU
  |
MaxPool1d (kernel_size=4)
  |
Conv1d (in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
  |
BatchNorm1d
  |
LeakyReLU
  |
Conv1d (in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
  |
BatchNorm1d
  |
LeakyReLU
  |
MaxPool1d (kernel_size=4)
  |
Dropout (p=0.1)
  |
Flatten
  |
Linear (in_features=3072, out_features=1024)
  |
LeakyReLU
  |
Linear (in_features=1024, out_features=128)
  |
LeakyReLU
  |
Linear (in_features=128, out_features=4)
  |
Softmax
Output


