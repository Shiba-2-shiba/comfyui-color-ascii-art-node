# sharpening.py
import logging
import torch
import torch.nn.functional as F

# --- Logger Setup ---
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3.Sharpening")
logger.setLevel(logging.INFO) # Default, will be adjusted by main node
if not logger.hasHandlers(): # Add basic handler if needed
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def unsharp_mask(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: float = 0.0) -> torch.Tensor:
    """
    Apply unsharp masking to a batch of images.
    Adapted from the provided unsharp-py.txt.

    Args:
        x (torch.Tensor): Input tensor with shape [B, C, H, W], range [0, 1].
        kernel_size (int): Size of the Gaussian kernel (should be odd).
        sigma (float): Standard deviation of the Gaussian kernel.
        amount (float): Strength of sharpening effect (1.0 = 100%).
        threshold (float): Minimum absolute difference to apply sharpening (range [0, 1]).

    Returns:
        torch.Tensor: Sharpened tensor with the same shape as input, clamped to [0, 1].
    """
    if x is None:
        raise ValueError("Input tensor 'x' cannot be None")
    if amount == 0:
        logger.debug("Unsharp mask amount is 0, skipping sharpening.")
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
        logger.warning(f"Kernel size must be odd, adjusted to {kernel_size}")

    channels = x.shape[1]
    device = x.device
    dtype = x.dtype

    logger.debug(f"Applying unsharp mask: kernel_size={kernel_size}, sigma={sigma}, amount={amount}, threshold={threshold}")

    try:
        # Create a 1D Gaussian kernel
        gauss_range = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=dtype)
        gauss = torch.exp(-(gauss_range ** 2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()

        # Convert it to a 2D kernel
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1) # [k, k]

        # Expand to match input channels [C, 1, k, k]
        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()

        # Apply padding to maintain spatial dimensions
        padding = kernel_size // 2

        # Create a blurred version of the image using depthwise convolution
        blurred = F.conv2d(x, kernel, padding=padding, groups=channels)

        # Calculate the mask (detail)
        mask = x - blurred

        # Apply threshold to the mask
        if threshold > 0:
            # Apply threshold only where the absolute difference exceeds the threshold
            mask = torch.where(torch.abs(mask) < threshold, torch.zeros_like(mask), mask)

        # Apply the sharpening: original + amount * details
        sharpened = x + amount * mask

        # Clamp values to maintain valid range [0, 1]
        return torch.clamp(sharpened, 0, 1)

    except Exception as e:
        logger.error(f"Error during unsharp masking: {e}", exc_info=True)
        # Return original image if sharpening fails
        return x
