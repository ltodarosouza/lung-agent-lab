import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generator"))

import models  # noqa: E402


class ChooseDeviceTest(unittest.TestCase):
    def test_auto_prefere_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(models.choose_device("auto").type, "cuda")

    def test_auto_usa_cpu_sem_acelerador(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(models.choose_device("auto").type, "cpu")

    def test_pedido_explicito_vence(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(models.choose_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
