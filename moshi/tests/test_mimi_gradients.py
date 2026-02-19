from huggingface_hub import hf_hub_download
import torch

from moshi.models import loaders, LMGen

mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device='cpu')
mimi.set_num_codebooks(8)


def test_mimi():
    # Instantiate a model
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    model = loaders.get_mimi(mimi_weight, device='cpu')
    model.set_num_codebooks(8)

    # Create a random audio signal
    x = torch.randn(2, 1, 16000, requires_grad=True)
    length = x.shape[-1]

    model.eval()
    # original implementation
    x_org_quantized = model.decode(model.encode(x))
    x_org_quantized = x_org_quantized[...,:length]


    for _ in range(10):
    
        x_quantized = model(x).x

        # Check that the audio reconstructions match
        assert torch.allclose(x_quantized, x_org_quantized, atol=1e-3)
        assert x_quantized.size() == x.size()

    loss = torch.pow(x_quantized, 2).sum()
    loss.backward()
    model.zero_grad()

    # Check that gradients are being tracked
    assert x.grad is not None

    # Check that gradient values look sensible
    print(x.grad)

    # Check that adding noise changes the gradients
    x_other = x.detach() + (0.1**0.5) * torch.randn(1, 1, 16000)
    x_other.requires_grad_()
    x_quantized_other = model(x_other).x
    loss_other = torch.pow(x_quantized_other, 2).sum()
    loss_other.backward()
    print(torch.mean(torch.abs(x.grad - x_other.grad)))


if __name__ == "__main__":
    test_mimi()
