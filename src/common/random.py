from common.imports import *

def pink_noise(*, height, width, channels=3, alpha=1.0, samples=None):
    """
    size: int, size of noise
    alpha: float, 1.0 is pink noise, 2.0 is brown noise

    return: torch.Tensor, [size]
    """
    # Generate white noise
    if samples is None:
        samples = torch.randn(height, width, channels)

    # Compute Fourier Transform
    spectrum = torch.fft.fftn(samples, dim=(0, 1))

    # Compute frequencies
    fy = torch.fft.fftfreq(height)[:, None]
    fx = torch.fft.fftfreq(width)[None, :]
    frequency = torch.sqrt(fx ** 2 + fy ** 2)
    frequency = frequency ** alpha

    frequency[0, 0] = 1.0

    spectrum = spectrum / frequency.unsqueeze(-1)
    samples = torch.fft.ifftn(spectrum, dim=(0, 1)).real

    # Normalize
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    return samples

def unit_rand3(batch_size=(), *, device=None):
    if isinstance(batch_size, int):
        batch_size = (batch_size,)
    phi = torch.rand(*batch_size, device=device) * (2 * math.pi)
    costheta = torch.rand(*batch_size, device=device) * 2 - 1

    theta = torch.acos(costheta)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def sample_with_prob(probs: Dict[Any, float], normalize=False):
    # probs: {item: prob}
    if normalize:
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
    cumprobs = np.cumsum(list(probs.values()))
    cumprobs = {k: v for k, v in zip(probs.keys(), cumprobs)}
    r = np.random.rand()
    for k, v in cumprobs.items():
        if r <= v:
            return k
    return None
