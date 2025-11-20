from torch import nn


class X2C(nn.Module):
    def __init__(self, num_concepts: int, 
                 x2c_model=None):
        super().__init__()
        if x2c_model is not None:
            self.model = nn.Sequential(x2c_model)
        else:
            raise ValueError("X2C model not specified")
        num_concepts = num_concepts
        self.model = nn.Sequential(x2c_model)

    def forward(self,x):
        return self.model(x) # outputs multiclass