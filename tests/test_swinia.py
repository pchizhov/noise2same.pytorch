import unittest
from typing import Optional, Tuple

import torch
import numpy as np
from parameterized import parameterized

from noise2same.backbone import swinia


class SwinIATestCase(unittest.TestCase):

    @parameterized.expand([
        ('1layer', 3, 4, 1, None),
        ('2layer', 3, 4, 2, None),
        ('3layer', 3, 4, 3, 5),
    ])
    def test_mlp(self, name: str, in_features: int, out_features: int, n_layers: int, hidden_features: Optional[int]):
        x = torch.randn(2, 8, 8, in_features)
        mlp = swinia.MLP(in_features, out_features, n_layers, hidden_features)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 8, 8, out_features))

    @parameterized.expand([
        ('vanilla', 16, (8, 8), (1, 1), 4, (4, 64, 1, 4)),
        ('shuffled', 16, (8, 8), (2, 2), 2, (2, 64, 4, 8)),
    ])
    def test_diag_window_attention(
            self,
            name: str,
            in_channels: int,
            window_size: Tuple[int, int],
            shuffle_group_size: Tuple[int, int],
            num_heads: int,
            expected_shape: Tuple[int, int, int, int]
    ):
        """
        Test expected shape of head partitioning and shuffle:
        (batch_size, num_heads, window_size_flat // shuffle_group_size_flat,
         shuffle_group_size_flat, in_channels // num_heads)

        Test identity of reassembled tensor.
        """
        window_size_flat = int(np.prod(window_size))
        shuffle_group_size_flat = int(np.prod(shuffle_group_size))
        x = torch.randn(2, window_size_flat * shuffle_group_size_flat, in_channels)

        dwa = swinia.DiagWinAttention(in_channels, window_size, shuffle_group_size, num_heads)
        partitioned = dwa.head_partition_and_shuffle(x)
        self.assertEqual(partitioned.shape[1:], expected_shape)
        reassembled = dwa.head_partition_and_shuffle_reversed(partitioned)
        self.assertTrue(torch.allclose(x, reassembled))

    def test_dot_product(self):
        attn = torch.randn(2, 64, 64)
        value = torch.randn(2, 64, 16)
        query = attn @ value
        self.assertEqual(query.shape, (2, 64, 16))
        query_einsum = torch.einsum('...qk,...kc->...qc', attn, value)
        self.assertEqual(query_einsum.shape, (2, 64, 16))
        self.assertTrue(torch.allclose(query, query_einsum))


if __name__ == '__main__':
    unittest.main()
