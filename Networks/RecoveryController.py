import torch
import torch.nn as nn

class RecoveryController(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size, clip_output, clip_min_value=0.0, clip_max_value=1.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        
        self.clip_output = clip_output
        self.clip_min_value = clip_min_value
        self.clip_max_value = clip_max_value

    def forward(self, x):
        x = self.lstm1(x)
        x = self.fc(x)
        if self.clip_output:
            x = torch.min(x, torch.ones_like(x)*self.clip_max_value)
            x = torch.max(x, torch.ones_like(x)*self.clip_min_value)
        return x

    