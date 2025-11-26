import torch
import torch.nn as nn

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gamma, model):
        """
        calculate orthogonality of convolutional layers
        input:
            model: current model
        """

        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        loss = 0.0
        for name, params in model.named_parameters():
            if ".conv" in name:
                w = params.view(-1,params.shape[0]) # flatten
                wwt = torch.matmul(w,w.T).to(device) # WW^T

                I = torch.eye(w.shape[0], requires_grad=True).to(device)
                
                wwt_ident = (wwt - I).to(device) # WW^T -I
                rand_vec = torch.rand(wwt_ident.shape[0],1, requires_grad=True).to(device)

                term1 = torch.matmul(wwt_ident,rand_vec) # z||WW^T - I||^2_2

                norm_t1 = torch.linalg.norm(term1,2) # 2-norm
                res1 = torch.div(wwt_ident,norm_t1)
                fin_res = torch.matmul(wwt_ident,res1)
                loss += torch.linalg.norm(fin_res,ord='fro')


        return gamma * loss
        
        return loss