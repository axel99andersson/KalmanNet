import torch
import torch.nn as nn

class RecoveryController(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size, clip_output, clip_min_value=0.0, clip_max_value=1.0, batch_size=30, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        
        self.clip_output = clip_output
        self.clip_min_value = clip_min_value
        self.clip_max_value = clip_max_value
        self.batch_size = batch_size

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x.reshape(self.batch_size, 1, -1)
        y = y.reshape(self.batch_size, 1, -1)
        x = torch.cat((x,y), dim=2)
        out, hidden = self.lstm1(x)
        x = self.fc(out[:,-1,:])
        if self.clip_output:
            x = torch.clip(x, -1.0, 1.0)
        return x

    