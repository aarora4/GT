import torch


def my_imfilter(image, filter):
    """
    Apply a filter to an image. Return the filtered image.
    Args
    - image: Torch tensor of shape (m, n, c)
    - filter: Torch tensor of shape (k, j)
    Returns
    - filtered_image: Torch tensor of shape (m, n, c)
    HINTS:
    - You may not use any libraries that do the work for you. Using torch to work
     with matrices is fine and encouraged. Using OpenCV or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take a long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Useful functions: torch.nn.functional.pad
    """
    filtered_image = torch.Tensor()

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    ffilter = filter.float()
    K, J = ffilter.size()
    M, N, C = image.size()
    R1 = int((K - 1)/2)
    R2 = int((K + 1)/2)
    S1 = int((J - 1)/2)
    S2 = int((J + 1)/2)
    
    filtered_image = torch.zeros(M, N, C).float()
    padded_signal = torch.zeros(M + K, N + J, C)
    padded_signal[R1:M + K - R2, S1:N + J - S2, :] = image
    
    for i in range(int(M)):
        for j in range(int(N)):
            filtered_image[i, j, :] += torch.einsum('kjc,kj->c', padded_signal[i:i + K, j: j + J, :].float(), ffilter)
    
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    return filtered_image


def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args
    - image1: Torch tensor of dim (m, n, c)
    - image2: Torch tensor of dim (m, n, c)
    - filter: Torch tensor of dim (x, y)
    Returns
    - low_frequencies: Torch tensor of shape (m, n, c)
    - high_frequencies: Torch tensor of shape (m, n, c)
    - hybrid_image: Torch tensor of shape (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are between
      0 and 1. This is known as 'clipping' ('clamping' in torch).
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    hybrid_image = torch.Tensor()
    low_frequencies = torch.Tensor()
    high_frequencies = torch.Tensor()

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    low_frequencies = my_imfilter(image1, filter)
    high_frequencies = image2 - my_imfilter(image2, filter)
    hybrid_image = torch.clamp(low_frequencies + high_frequencies, 0, 1)
    

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    return low_frequencies, high_frequencies, hybrid_image
