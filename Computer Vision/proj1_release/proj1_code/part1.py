import torch


def my_1dfilter(signal: torch.FloatTensor,
                kernel: torch.FloatTensor) -> torch.FloatTensor:
    """Filters the signal by the kernel.

    output = signal * kernel where * denotes the cross-correlation function.
    Cross correlation is similar to the convolution operation with difference
    being that in cross-correlation we do not flip the sign of the kernel.

    Reference: 
    - https://mathworld.wolfram.com/Cross-Correlation.html
    - https://mathworld.wolfram.com/Convolution.html

    Note:
    1. The shape of the output should be the same as signal.
    2. You may use zero padding as required. Please do not use any other 
       padding scheme for this function.
    3. Take special care that your function performs the cross-correlation 
       operation as defined even on inputs which are asymmetric.

    Args:
        signal (torch.FloatTensor): input signal. Shape=(N,)
        kernel (torch.FloatTensor): kernel to filter with. Shape=(K,)

    Returns:
        torch.FloatTensor: filtered signal. Shape=(N,2)
    """
    filtered_signal = torch.FloatTensor()

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    K = kernel.size(0)
    N = signal.size(0)
    if K % 2 == 1:
        R1 = int((K - 1)/2)
        R2 = int((K + 1)/2)
    else:
        R1 = int(K / 2)
        R2 = int(K / 2)
    
    filtered_signal = torch.zeros(N)
    padded_signal = torch.zeros(N + K)
    padded_signal[R1:N + K - R2] = signal
    for i in range(N):
        filtered_signal[i] += torch.dot(padded_signal[i:i + K], kernel)
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return filtered_signal
