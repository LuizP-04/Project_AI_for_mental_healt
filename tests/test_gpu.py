"""GPU tests. They run only where PyTorch can use CUDA, so the CI skips them."""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU here")


def test_fp16_matrix_product_stays_close_to_fp32():
    # The classifiers will run in FP16 to fit in 6 GB; results must stay close to FP32.
    torch.manual_seed(0)
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    expected = a @ b
    result = (a.half() @ b.half()).float()
    torch.testing.assert_close(result, expected, rtol=2e-2, atol=0.5)
