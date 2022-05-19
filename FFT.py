import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


def divide_spectrum(trg_img):

    fft_trg_np = torch.fft.fft2( trg_img, dim=(-3, -2) )
    amp_target, pha_trg = torch.abs(fft_trg_np), torch.angle(fft_trg_np)

    return amp_target, pha_trg



def amp_spectrum_swap( amp_local, amp_target, L=1, ratio=0.5):
    
    a_local = torch.fft.fftshift( amp_local, dim=(-3, -2) ) #put zeros in the center of each channel
    a_trg = torch.fft.fftshift( amp_target, dim=(-3, -2) )

    _, cha, h, w = a_local.shape
    b = (np.floor(np.amin((h,w))*L) / 2  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b
    w1 = c_w-b
    w2 = c_w+b

    a_local[:,:,h1:h2,w1:w2] = a_local[:,:,h1:h2,w1:w2] * ratio + a_trg[:,:,h1:h2,w1:w2] * (1- ratio)
    a_local = torch.fft.ifftshift( a_local, dim=(-3, -2) ) #shift zero back to the original space
    return a_local



def pha_spectrum_swap(pha_local, pha_target, L=1, ratio=0.5):
    
    a_local = torch.fft.fftshift( pha_local, dim=(-3, -2) ) #put zeros in the center of each channel
    a_trg = torch.fft.fftshift( pha_target, dim=(-3, -2) )

    _, cha, h, w = a_local.shape
    b = (np.floor(np.amin((h,w))*L) / 2  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b
    w1 = c_w-b
    w2 = c_w+b

    a_local[:,:,h1:h2,w1:w2] = a_local[:,:,h1:h2,w1:w2] * ratio + a_trg[:,:,h1:h2,w1:w2] * (1- ratio)
    a_local = torch.fft.ifftshift( a_local, dim=(-3, -2) ) #shift zero back to the original space
    return a_local



def freq_space_interpolation( local_img, amp_target, L=0 , ratio=0):
    
    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = torch.fft.fft2( local_img_np, dim=(-3, -2) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = torch.abs(fft_local_np), torch.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * torch.exp( 1j * pha_local )
    local_in_trg = torch.fft.ifft2( fft_local_, dim=(-3, -2) )
    local_in_trg = torch.real(local_in_trg)
    
    return local_in_trg


def pha_space_interpolation( local_img, pha_target, L=0 , ratio=0):
    
    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = torch.fft.fft2( local_img_np, dim=(-3, -2) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = torch.abs(fft_local_np), torch.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    pha_local_ = pha_spectrum_swap( pha_local, pha_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local * torch.exp( 1j * pha_local_ )
    local_in_trg = torch.fft.ifft2( fft_local_, dim=(-3, -2) )
    local_in_trg = torch.real(local_in_trg)
    
    return local_in_trg
