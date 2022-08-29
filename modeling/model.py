import torch
import torch.nn as nn

class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            #        (in_channels,out_channels,kernel_size) rafael
            nn.Conv1d(input_size, 64, 5),  #-4
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5), #-4
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5), #-4
            nn.ReLU(),
            nn.MaxPool1d(2),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
        	nn.Dropout(),
        	nn.Linear(3456, 128),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(128, num_classes),
        	)

    def forward(self, x):
    	x = self.features(x)
    	x = x.view(x.size(0), 3456)
    	out = self.classifier(x)

    	return out