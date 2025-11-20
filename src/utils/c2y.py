from torch import nn


class C2Y(nn.Module):
    def __init__(self, num_classes: int, 
                 num_concepts: int, 
                 c2y_model=None):
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        super().__init__()
        if c2y_model == None:
            c2y_model = nn.Sequential(*[nn.Linear(self.num_concepts, self.num_classes)])
        else:
            c2y_model = c2y_model

        self.model = c2y_model

    def forward(self,x):
        return self.model(x) # outputs 1 class