import torch
from packaging import version
__all__ = ['AverageMeter', 'evaluate_nmse','evaluate_nmse2', 'evaluate_rho','evaluate_nmse_mag', 'evaluate_rho_mag','post_processing']


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def evaluate_nmse(Hi, Hi_hat):
    r""" Evaluation of decoding implemented in PyTorch Tensor 
    Computes normalized mean square error (NMSE)"""
    #print("testing again")
    with torch.no_grad():
        Hi_real = torch.reshape(Hi[:, 0, :, :], (Hi.size(0), -1))
        Hi_imag = torch.reshape(Hi[:, 1, :, :], (Hi.size(0), -1))
        Hi_comp = (Hi_real-0.5) + 1j * (Hi_imag-0.5)

        Hi_hat_real = torch.reshape(Hi_hat[:, 0, :, :], (Hi_hat.size(0), -1))
        Hi_hat_imag = torch.reshape(Hi_hat[:, 1, :, :], (Hi_hat.size(0), -1))
        Hi_hat_comp = (Hi_hat_real-0.5) + 1j * (Hi_hat_imag-0.5)

        mse = torch.sum(torch.square(abs(Hi_comp - Hi_hat_comp)), axis = 1)
        # print(torch.mean(mse))
        power = torch.sum(torch.square(abs(Hi_comp)), axis = 1)
        # print(torch.mean(power))

        nmse = 10 * torch.log10(torch.mean(mse / power))
        #sig_pow =(power_gt.sum(dim=[1,2])).mean()      
        return nmse
    
def evaluate_nmse2(Hi, Hi_hat):
    r""" Evaluation of decoding implemented in PyTorch Tensor 
    Computes normalized mean square error (NMSE)"""
    #print("testing again")
    with torch.no_grad():
        Hi_real = torch.reshape(Hi[:, 0, :, :], (Hi.size(0), -1))
        Hi_imag = torch.reshape(Hi[:, 1, :, :], (Hi.size(0), -1))
        Hi_comp = Hi_real + 1j * Hi_imag

        Hi_hat_real = torch.reshape(Hi_hat[:, 0, :, :], (Hi_hat.size(0), -1))
        Hi_hat_imag = torch.reshape(Hi_hat[:, 1, :, :], (Hi_hat.size(0), -1))
        Hi_hat_comp = Hi_hat_real + 1j * Hi_hat_imag

        mse = torch.sum(torch.square(abs(Hi_comp - Hi_hat_comp)), axis = 1)
        power = torch.sum(torch.square(abs(Hi_comp)), axis = 1)
        print(torch.mean(mse))
        nmse = 10 * torch.log10(torch.mean(mse / power))
        #sig_pow =(power_gt.sum(dim=[1,2])).mean()      
        return nmse
    
    

def evaluate_rho(raw_pred, raw_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor Computes rho."""

    with torch.no_grad():               
        norm_pred = raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2
        norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        norm_gt = raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2
        norm_gt = torch.sqrt(norm_gt.sum(dim=1))

        real_cross = raw_pred[..., 0] * raw_gt[..., 0] + raw_pred[..., 1] * raw_gt[..., 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = raw_pred[..., 0] * raw_gt[..., 1] - raw_pred[..., 1] * raw_gt[..., 0]
        imag_cross = imag_cross.sum(dim=1)
        norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)

        rho = (norm_cross / (norm_pred * norm_gt)).mean()

        return rho
    

    
def evaluate_nmse_mag(sparse_pred, sparse_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor 
    Computes normalized mean square error (NMSE)    """

    with torch.no_grad():
        sparse_gt =sparse_gt
        sparse_pred=sparse_pred
       
        # Calculate the NMSE
        power_gt = sparse_gt ** 2 
        difference = sparse_gt - sparse_pred
        mse = difference** 2 
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
        return  mse,nmse

def evaluate_rho_mag(raw_pred, raw_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor Computes rho."""

    with torch.no_grad():
               
        norm_pred = raw_pred** 2 
        norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        norm_gt = raw_gt**2 
        norm_gt = torch.sqrt(norm_gt.sum(dim=1))

        real_cross = raw_pred* raw_gt
        real_cross = real_cross.sum(dim=1)
        norm_cross = torch.sqrt(real_cross ** 2)

        rho = (norm_cross / (norm_pred * norm_gt)).mean()

        return rho
    
def post_processing(sparse_pred):
        r""" Evaluation of decoding implemented in PyTorch Tensor Computes rho."""

        with torch.no_grad():
            # Basic params
            nt = 32
            nc = 32
            nc_expand = 257

            # De-centralize
            sparse_pred = sparse_pred
         
            # Calculate the Rho
            n = sparse_pred.size(0)
            sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # Move the real/imaginary dim to the last
            zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
            # When pytorch version is above 1.7.0, complex number representation is changed from [a, b] to [a, b.j]
            if version.parse(torch.__version__) > version.parse("1.7.0"):
                sparse_pred = torch.view_as_complex(torch.cat((sparse_pred, zeros), dim=2))
                raw_pred = torch.view_as_real(torch.fft.fft(sparse_pred))[:, :, :125, :]
            else:
                sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
                raw_pred = torch.fft(sparse_pred, signal_ndim=1)[:, :, :125, :]
            
            return raw_pred
