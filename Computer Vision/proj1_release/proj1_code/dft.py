import torch
import torch.nn as nn
import torch.nn.functional as F


def DFT_matrix(N):
    """
    Takes the square matrix dimension as input, generate the DFT matrix correspondingly
    Args
    - N: the DFT matrix dimension
    Returns
    - U: the generated DFT matrix (torch.Tensor) of size N*N; it should be a numpy matrix with data type as 'complex'
    """
    U = torch.Tensor()

    torch.pi = torch.acos(torch.zeros(1)).item() * \
        2  # which is 3.1415927410125732
        
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    U = torch.zeros(N, N, 2)
    
    range = torch.arange(N).float()
    product_range = 2 * torch.pi * torch.einsum('i,j->ij', range, range) / N
    
    U[:, :, 0] = torch.cos(product_range) / N
    U[:, :, 1] = -torch.sin(product_range) / N
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return U


def Compl_mul_Real(m1, m2):
    """
    Takes the one complex tensor matrix and a real matrix, and do matrix multiplication
    Args
    - m1: the Tensor matrix (m,n,2) which represents complex number;
    E.g., the real part is t1[:,:,0], the imaginary part is t1[:,:,1]
    - m2: the real matrix (m,n)
    Returns
    - U: matrix multiplication result in the same form as input
    """
    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2
    imag2 = torch.zeros(real2.shape)
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


def Compl_mul_Compl(m1, m2):
    """
    Takes the two complex tensor matrix and do matrix multiplication
    Args
    - t1, t2: the Tensor matrix (m,n,2) which represents complex number;
    E.g., the real part is t1[:,:,0], the imaginary part is t1[:,:,1]
    Returns
    - U: matrix multiplication result in the same form as input
    """
    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2[:, :, 0]
    imag2 = m2[:, :, 1]
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


def my_dft(img):
    """
    Takes an square image as input, performs Discrete Fourier Transform for the image matrix
    This function is expected to behave similar as torch.rfft(x,2,onesided=False)
    Args
    - img: a 2D grayscale image (torch.Tensor) whose width equals height,
    Returns
    - dft: the DFT results of img
    - hints: we provide two function to do complex matrix multiplication:
    Compl_mul_Real and Compl_mul_Compl
    """
    dft = torch.Tensor()

    assert img.shape[0] == img.shape[1], "Input image should be a square matrix"

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    _, N = img.size()
    kernel = DFT_matrix(N)
    dft = torch.zeros(N, N, 2)
    
    if N % 2 == 1:
        R1 = int((N - 1)/2)
        R2 = int((N + 1)/2)
    else:
        R1 = int(N / 2)
        R2 = int((N - 1) / 2)
    
    padded_signal = torch.zeros(N + R1 + R2, N + R1 + R2)
    padded_signal[R1:N + R1, R1:N + R1] = img
    padded_signal = padded_signal.float().view(1, 1, N + R1 + R2, N + R1 + R2)
    print(padded_signal.shape, kernel.shape)
    dft[:,:, 0] = F.conv2d(padded_signal, kernel.float()[:, :, 0].view(1, 1, N, N), groups = 1)
    dft[:,:, 1] = F.conv2d(padded_signal, kernel.float()[:, :, 1].view(1, 1, N, N), groups = 1)

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return dft


def dft_filter(img):
    """
    Takes an square image as input, performs a low-pass filter and return the filtered image
    Args
    - img: a 2D grayscale image whose width equals height
    Returns
    - img_back: the filtered image
    Hints
    - You will need your implemented DFT filter for this function
    - Shifting the DFT results makes your filter easier
    - We don't care how much frequency you want to retain, if only it returns reasonable results
    - Since you already implemented DFT part, you're allowed to use the torch.ifft in this part for convenience, though not necessary
    """

    img_back = torch.Tensor()

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    _ , N = img.size()
    if N % 2 == 1:
        R1 = int((N - 1)/2)
        R2 = int((N + 1)/2)
    else:
        R1 = int(N / 2)
        R2 = int((N - 1) / 2)
        
    dft = my_dft(img)
    filtered_dft = torch.zeros(N + R1 + R2, N + R1 + R2, 2)
    filtered_dft[R1:N + R1, R1:N + R1] = dft
    
    kernel = DFT_matrix(N)
    kernel[:, :, 1] = -kernel[:, :, 1]
    
    img_back = torch.zeros(N, N)
    
    img_back = F.conv2d(filtered_dft.float()[:, :, 0].view(1, 1, N + R1 + R2, N + R1 + R2), kernel.float()[:, :, 0].view(1, 1, N, N), groups = 1)
    img_back -= F.conv2d(filtered_dft.float()[:, :, 1].view(1, 1, N + R1 + R2, N + R1 + R2), kernel.float()[:, :, 1].view(1, 1, N, N), groups = 1)

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return img_back
