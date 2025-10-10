import torch.nn as nn

class HTRModel(nn.Module):
    def __init__(self, num_classes: int):
        super(HTRModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(
            input_size=128 * 16,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.contiguous().view(b, w, c * h)

        rnn_out, _ = self.rnn(conv)
        out = self.fc(rnn_out)
        return out
