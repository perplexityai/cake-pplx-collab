"""
Copyright (c) 2026, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Program selection by token count of the Cake BF16 RMSNorm training kernels
(SM100 / SM103 / SM107), rendered from the registry of this package
(:mod:`.cake_jit`) by its exporter; do not edit.  ``PROGRAM_WINDOWS`` is the
table ``cake_jit.program_windows`` derives from the registry's ``MODULES`` and
``ROUTES`` at call time; ``EDGES`` the measured outcome at every window edge;
``NOTES`` the contract of the outputs across the windows; ``SECTIONS`` the text
the host module appends to the forward and backward docstrings;
``ACCURACY_MIN_SMS`` and ``ACCURACY_EDGES`` the backward accuracy guard (see
``NOTES``).  The package tests hold the table to the registry and check the
outputs bitwise across every ``same`` edge.
"""

from __future__ import annotations

# Per architecture and route (``cake_jit.route_key``): the windows ``above < T <= at_most``
# (``at_most`` None: unbounded) and the program each launches, from the smallest rows up.
PROGRAM_WINDOWS: dict[str, dict[str, tuple[tuple[int, int | None, str], ...]]] = {
    "sm_100a": {
        "bwd_h1024": (
            (0, 1, "cake_rmsnorm_det_train_2dc3fb268514e9182c6a"),
            (1, 64, "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2"),
            (64, 128, "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac"),
            (128, 1024, "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2"),
            (1024, 8192, "cake_rmsnorm_det_train_0c581a133624ff396ab4"),
            (8192, None, "cake_rmsnorm_det_train_acd331db1201bf2fb44e"),
        ),
        "bwd_w32_h1024": (
            (0, 1, "cake_rmsnorm_det_train_153f811937bb0157224b"),
            (1, 1024, "cake_rmsnorm_det_train_744d74299c2d45867c94"),
            (1024, 8192, "cake_rmsnorm_det_train_561bea88a25c2f245c29"),
            (8192, None, "cake_rmsnorm_det_train_b201ba67ac366efbd7be"),
        ),
        "bwd_h128": (
            (0, 1024, "cake_rmsnorm_det_train_9346600882e53a589509"),
            (1024, 2048, "cake_rmsnorm_det_train_401a018c68bc082078b2"),
            (2048, 11201, "cake_rmsnorm_det_train_d197bcc34403e94ee2bd"),
            (11201, None, "cake_rmsnorm_det_train_401a018c68bc082078b2"),
        ),
        "bwd_w32_h128": (
            (0, 1024, "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e"),
            (1024, 2048, "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea"),
            (2048, 11201, "cake_rmsnorm_det_train_e23083c7af388f9e2f50"),
            (11201, None, "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea"),
        ),
        "bwd_h1280": (
            (0, 1, "cake_rmsnorm_det_train_8895b0224bb3d596a3a6"),
            (1, 128, "cake_rmsnorm_det_train_a5f2942d61e591f57a76"),
            (128, 512, "cake_rmsnorm_det_train_809235a3965d337a6877"),
            (512, 4096, "cake_rmsnorm_det_train_12bbedff93fcaafa475d"),
            (4096, None, "cake_rmsnorm_det_train_72a918b00d72b017830c"),
        ),
        "bwd_w32_h1280": (
            (0, 1, "cake_rmsnorm_det_train_103805d93149c8668264"),
            (1, 128, "cake_rmsnorm_det_train_8d8ef930b7d650c5538a"),
            (128, 512, "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d"),
            (512, 4096, "cake_rmsnorm_det_train_80956d258172dfa76c2a"),
            (4096, None, "cake_rmsnorm_det_train_ef629c30e75a5fee7667"),
        ),
        "bwd_h1536": (
            (0, 1, "cake_rmsnorm_det_train_f6233ff84d0b93fe45db"),
            (1, 768, "cake_rmsnorm_det_train_4176a24f8ee981dd2bb2"),
            (768, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 1184, "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe"),
            (1184, 161792, "cake_rmsnorm_det_train_38e32d90f62f7cfc7b51"),
            (161792, None, "cake_rmsnorm_det_train_b457c4d134e44476dc6f"),
        ),
        "bwd_w32_h1536": (
            (0, 1, "cake_rmsnorm_det_train_5b177296918aca760fa2"),
            (1, 768, "cake_rmsnorm_det_train_06bbfeb2e313d41451fa"),
            (768, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 1184, "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b"),
            (1184, 161792, "cake_rmsnorm_det_train_e30c1899e859da4f4c68"),
            (161792, None, "cake_rmsnorm_det_train_8777b1c8538527c9aaf7"),
        ),
        "bwd_h2048": (
            (0, 1, "cake_rmsnorm_det_train_b13b0c73ece62938899a"),
            (1, 768, "cake_rmsnorm_det_train_923e668c53f60f0b256b"),
            (768, 1024, "cake_rmsnorm_det_train_cc26e316ae40d683a118"),
            (1024, 2047, "cake_rmsnorm_det_train_c73074cf44f07fcf3bb6"),
            (2047, 22528, "cake_rmsnorm_det_train_cc26e316ae40d683a118"),
            (22528, 58880, "cake_rmsnorm_det_train_ceeef043ac64f37fc529"),
            (58880, None, "cake_rmsnorm_det_train_ceeef043ac64f37fc529"),
        ),
        "bwd_w32_h2048": (
            (0, 1, "cake_rmsnorm_det_train_d22b3e8ce96e49997448"),
            (1, 768, "cake_rmsnorm_det_train_9030dbeb948761ec2258"),
            (768, 1024, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (1024, 16384, "cake_rmsnorm_det_train_b187e5989302def22b2d"),
            (16384, 58880, "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9"),
            (58880, None, "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9"),
        ),
        "bwd_h256": (
            (0, 87, "cake_rmsnorm_det_train_6959ea32516c82acae98"),
            (87, 2048, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (2048, 5120, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (5120, 12288, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (12288, 262144, "cake_rmsnorm_det_train_6bdbfe619b8270ff9a21"),
            (262144, 1572864, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (1572864, None, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
        ),
        "bwd_w32_h256": (
            (0, 87, "cake_rmsnorm_det_train_c484e2676d7b535ff396"),
            (87, 2048, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (2048, 5120, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (5120, 12288, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (12288, 262144, "cake_rmsnorm_det_train_6554d4eb4362141cb058"),
            (262144, 1572864, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (1572864, None, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
        ),
        "bwd_h4096": (
            (0, 203, "cake_rmsnorm_det_train_4822793ea0c954c11af6"),
            (203, 256, "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9"),
            (256, 1024, "cake_rmsnorm_det_train_59270fb249dcba1ff679"),
            (1024, 22528, "cake_rmsnorm_det_train_5572b2671862fe670ef4"),
            (22528, 32768, "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a"),
            (32768, 44032, "cake_rmsnorm_det_train_35a0b0ca6b2060313255"),
            (44032, None, "cake_rmsnorm_det_train_35a0b0ca6b2060313255"),
        ),
        "bwd_w32_h4096": (
            (0, 203, "cake_rmsnorm_det_train_8616070567821eaa3e35"),
            (203, 256, "cake_rmsnorm_det_train_20c64c610d22c6db4c05"),
            (256, 1024, "cake_rmsnorm_det_train_c9f55c0ae4b198c859d1"),
            (1024, 22528, "cake_rmsnorm_det_train_47a01388444960da7aae"),
            (22528, 32768, "cake_rmsnorm_det_train_a8eb754739a1589de3ec"),
            (32768, 44032, "cake_rmsnorm_det_train_626aed743358c27513f1"),
            (44032, None, "cake_rmsnorm_det_train_626aed743358c27513f1"),
        ),
        "bwd_h512": (
            (0, 1, "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c"),
            (1, 227, "cake_rmsnorm_det_train_931757c844a9d257c08b"),
            (227, 1024, "cake_rmsnorm_det_train_09601d42628de24f6f87"),
            (1024, 4096, "cake_rmsnorm_det_train_123db0b2f7fb38cfab03"),
            (4096, 32768, "cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036"),
            (32768, None, "cake_rmsnorm_det_train_8877322a3002e3a4fefd"),
        ),
        "bwd_w32_h512": (
            (0, 1, "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a"),
            (1, 227, "cake_rmsnorm_det_train_404c8f54ab3129877350"),
            (227, 1024, "cake_rmsnorm_det_train_9f07cea1d13b923ed91e"),
            (1024, 4096, "cake_rmsnorm_det_train_632ca6ff01848a71f9f2"),
            (4096, 32768, "cake_rmsnorm_det_train_1f628cf4fd17b4ea9441"),
            (32768, None, "cake_rmsnorm_det_train_659bfc53e5372850ae2e"),
        ),
        "bwd_h5120": (
            (0, 1, "cake_rmsnorm_det_train_080e0a6c3b18e9bad790"),
            (1, 256, "cake_rmsnorm_det_train_dd3a35731e2e1665dd71"),
            (256, 36864, "cake_rmsnorm_det_train_bfde9b88e800fc34722c"),
            (36864, 50688, "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c"),
            (50688, None, "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c"),
        ),
        "bwd_w32_h5120": (
            (0, 1, "cake_rmsnorm_det_train_747d86f34cb58479c7ca"),
            (1, 256, "cake_rmsnorm_det_train_747d86f34cb58479c7ca"),
            (256, 36864, "cake_rmsnorm_det_train_f9b2d42c4fc7a5a1aa48"),
            (36864, 50688, "cake_rmsnorm_det_train_9479b1298a1b19d4d996"),
            (50688, None, "cake_rmsnorm_det_train_9479b1298a1b19d4d996"),
        ),
        "bwd_h6144": (
            (0, 384, "cake_rmsnorm_det_train_f2271417b1a596d0540f"),
            (384, 4096, "cake_rmsnorm_det_train_61421bc72e57d14d8a32"),
            (4096, 42496, "cake_rmsnorm_det_train_b93bf060d4d3defc8150"),
            (42496, None, "cake_rmsnorm_det_train_b93bf060d4d3defc8150"),
        ),
        "bwd_w32_h6144": (
            (0, 384, "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5"),
            (384, 4096, "cake_rmsnorm_det_train_41b80b6af97974feb181"),
            (4096, 42496, "cake_rmsnorm_det_train_62beb52fa50cc8624c9f"),
            (42496, None, "cake_rmsnorm_det_train_62beb52fa50cc8624c9f"),
        ),
        "bwd_residual_h6144": (
            (0, 1152, "cake_rmsnorm_det_train_864e84f3d874f850c091"),
            (1152, 42496, "cake_rmsnorm_det_train_c1a3be946e94e94233af"),
            (42496, None, "cake_rmsnorm_det_train_84385468c8e8ef19f3dc"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 1152, "cake_rmsnorm_det_train_71d36617003ab7742a77"),
            (1152, 42496, "cake_rmsnorm_det_train_ab0e20feefa296ffff3c"),
            (42496, None, "cake_rmsnorm_det_train_ab0e20feefa296ffff3c"),
        ),
        "fwd_h1024": (
            (0, 2, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (2, 4, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
            (4, 512, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (512, 1024, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
            (1024, 2048, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (2048, 4096, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
            (4096, 32768, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (32768, None, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
        ),
        "fwd_w32_h1024": (
            (0, 2, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (2, 4, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (4, 32, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (32, 64, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (64, 512, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (512, 1024, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (1024, 32768, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (32768, None, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
        ),
        "fwd_h128": (
            (0, 2, "cake_rmsnorm_det_train_3808022dd402387eed16"),
            (2, 4, "cake_rmsnorm_det_train_628267fc724c2c8f0e07"),
            (4, 16384, "cake_rmsnorm_det_train_3808022dd402387eed16"),
            (16384, None, "cake_rmsnorm_det_train_628267fc724c2c8f0e07"),
        ),
        "fwd_w32_h128": (
            (0, 2, "cake_rmsnorm_det_train_513f4ffbfefd9d019863"),
            (2, 4, "cake_rmsnorm_det_train_1ecf9c260e418bc70046"),
            (4, 64, "cake_rmsnorm_det_train_513f4ffbfefd9d019863"),
            (64, 128, "cake_rmsnorm_det_train_1ecf9c260e418bc70046"),
            (128, 16384, "cake_rmsnorm_det_train_513f4ffbfefd9d019863"),
            (16384, None, "cake_rmsnorm_det_train_1ecf9c260e418bc70046"),
        ),
        "fwd_h1280": (
            (0, 16, "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5"),
            (16, 32, "cake_rmsnorm_det_train_03a74dfe62df95c2a752"),
            (32, 256, "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5"),
            (256, 512, "cake_rmsnorm_det_train_03a74dfe62df95c2a752"),
            (512, 8192, "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5"),
            (8192, 24576, "cake_rmsnorm_det_train_39b9b41b16769252153d"),
            (24576, 32768, "cake_rmsnorm_det_train_007e82b24390c88156ac"),
            (32768, None, "cake_rmsnorm_det_train_03b5516039c533b453b9"),
        ),
        "fwd_w32_h1280": (
            (0, 1, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (1, 4, "cake_rmsnorm_det_train_c43709cadf31ed04033c"),
            (4, 8, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (8, 16, "cake_rmsnorm_det_train_c43709cadf31ed04033c"),
            (16, 256, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (256, 1024, "cake_rmsnorm_det_train_c43709cadf31ed04033c"),
            (1024, 2048, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (2048, 32768, "cake_rmsnorm_det_train_58c2fce1ea0b40758e91"),
            (32768, None, "cake_rmsnorm_det_train_99b209a37f0203af57ce"),
        ),
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_det_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_det_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_det_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (1, 1024, "cake_rmsnorm_det_train_a5f323db12acfb2cd565"),
            (1024, 2048, "cake_rmsnorm_det_train_780b9833fd0a236ef0e4"),
            (2048, 4096, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_det_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (1, 1024, "cake_rmsnorm_det_train_27fd540856828056457a"),
            (1024, 2048, "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b"),
            (2048, 4096, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 1024, "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8"),
            (1024, 8192, "cake_rmsnorm_det_train_0489c784c13bb25499f6"),
            (8192, 40960, "cake_rmsnorm_det_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_det_train_1388ab779a35c4e41efb"),
            (114688, 196608, "cake_rmsnorm_det_train_466331ff3cd2d6c62738"),
            (196608, 327680, "cake_rmsnorm_det_train_ed9483d8c356affee55d"),
            (327680, None, "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 1024, "cake_rmsnorm_det_train_f8087de00e0409186d93"),
            (1024, 8192, "cake_rmsnorm_det_train_f2e721db4c30900f12fe"),
            (8192, 40960, "cake_rmsnorm_det_train_77456435c9fbd1d18fcf"),
            (40960, 81920, "cake_rmsnorm_det_train_42d82d26e97c36651dae"),
            (81920, 114688, "cake_rmsnorm_det_train_b0299f6e0881fb35f27b"),
            (114688, 196608, "cake_rmsnorm_det_train_32bebe5a7782311d60a0"),
            (196608, 327680, "cake_rmsnorm_det_train_d0a656f053396a58c49a"),
            (327680, None, "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_det_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
            (1024, 6144, "cake_rmsnorm_det_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_det_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
            (1024, 6144, "cake_rmsnorm_det_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_det_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_det_train_7673978dc698790054d2"),
            (4096, 24576, "cake_rmsnorm_det_train_3d20ab4a432bddf7349a"),
            (24576, None, "cake_rmsnorm_det_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_det_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_det_train_e2eabcb3b0915d916a13"),
            (4096, 24576, "cake_rmsnorm_det_train_7da36ff557bb2bac8d89"),
            (24576, None, "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 64, "cake_rmsnorm_det_train_6e12e07353014329ebad"),
            (64, 640, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_det_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 64, "cake_rmsnorm_det_train_c1819da87ab45ebb8d56"),
            (64, 640, "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e"),
            (640, 6144, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
            (6144, 16384, "cake_rmsnorm_det_train_b49de9b47372fa4e1051"),
            (16384, None, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
        ),
        "fwd_h6144": (
            (0, 768, "cake_rmsnorm_det_train_bad4db9153cc903b7ace"),
            (768, 1024, "cake_rmsnorm_det_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_det_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 768, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (768, 1024, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d"),
            (6144, None, "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_8b4730338cac98ec272d"),
            (256, 1024, "cake_rmsnorm_det_train_03bdf5667404141e1eaf"),
            (1024, 4096, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
            (4096, None, "cake_rmsnorm_det_train_41805c5801f6956b4ce4"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_3e55e44ce21eff7d8931"),
            (256, 1024, "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485"),
            (1024, 4096, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
            (4096, None, "cake_rmsnorm_det_train_4be7a14e78ff59fa96e2"),
        ),
    },
    "sm_103a": {
        "bwd_h1024": (
            (0, 1, "cake_rmsnorm_det_train_2dc3fb268514e9182c6a"),
            (1, 64, "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2"),
            (64, 128, "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac"),
            (128, 1024, "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2"),
            (1024, 1536, "cake_rmsnorm_det_train_5e32dd889d72a75e348b"),
            (1536, 2048, "cake_rmsnorm_det_train_5e32dd889d72a75e348b"),
            (2048, 4096, "cake_rmsnorm_det_train_5e32dd889d72a75e348b"),
            (4096, 8192, "cake_rmsnorm_det_train_5e32dd889d72a75e348b"),
            (8192, 12288, "cake_rmsnorm_det_train_5e32dd889d72a75e348b"),
            (12288, None, "cake_rmsnorm_det_train_acd331db1201bf2fb44e"),
        ),
        "bwd_w32_h1024": (
            (0, 1, "cake_rmsnorm_det_train_153f811937bb0157224b"),
            (1, 1024, "cake_rmsnorm_det_train_744d74299c2d45867c94"),
            (1024, 1536, "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca"),
            (1536, 2048, "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca"),
            (2048, 12288, "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca"),
            (12288, None, "cake_rmsnorm_det_train_b201ba67ac366efbd7be"),
        ),
        "bwd_h128": (
            (0, 1024, "cake_rmsnorm_det_train_9346600882e53a589509"),
            (1024, 2048, "cake_rmsnorm_det_train_401a018c68bc082078b2"),
            (2048, 12288, "cake_rmsnorm_det_train_d197bcc34403e94ee2bd"),
            (12288, None, "cake_rmsnorm_det_train_401a018c68bc082078b2"),
        ),
        "bwd_w32_h128": (
            (0, 1024, "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e"),
            (1024, 2048, "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea"),
            (2048, 12288, "cake_rmsnorm_det_train_e23083c7af388f9e2f50"),
            (12288, None, "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea"),
        ),
        "bwd_h1280": (
            (0, 1, "cake_rmsnorm_det_train_8895b0224bb3d596a3a6"),
            (1, 128, "cake_rmsnorm_det_train_a5f2942d61e591f57a76"),
            (128, 512, "cake_rmsnorm_det_train_809235a3965d337a6877"),
            (512, 2048, "cake_rmsnorm_det_train_12bbedff93fcaafa475d"),
            (2048, None, "cake_rmsnorm_det_train_c646e0976e08c6fe8a91"),
        ),
        "bwd_w32_h1280": (
            (0, 1, "cake_rmsnorm_det_train_103805d93149c8668264"),
            (1, 128, "cake_rmsnorm_det_train_8d8ef930b7d650c5538a"),
            (128, 512, "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d"),
            (512, 4096, "cake_rmsnorm_det_train_80956d258172dfa76c2a"),
            (4096, None, "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4"),
        ),
        "bwd_h1536": (
            (0, 1, "cake_rmsnorm_det_train_e254714a110f4d45ea02"),
            (1, 31, "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80"),
            (31, 214, "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80"),
            (214, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 1216, "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe"),
            (1216, 214016, "cake_rmsnorm_det_train_38e32d90f62f7cfc7b51"),
            (214016, None, "cake_rmsnorm_det_train_9cd3c766db4b70ed7213"),
        ),
        "bwd_w32_h1536": (
            (0, 1, "cake_rmsnorm_det_train_9f139a52a93ccc69c3d1"),
            (1, 31, "cake_rmsnorm_det_train_698c3d36d67408394ccf"),
            (31, 214, "cake_rmsnorm_det_train_698c3d36d67408394ccf"),
            (214, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 1216, "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b"),
            (1216, 214016, "cake_rmsnorm_det_train_e30c1899e859da4f4c68"),
            (214016, None, "cake_rmsnorm_det_train_51f46ff164b142ec8888"),
        ),
        "bwd_h2048": (
            (0, 1, "cake_rmsnorm_det_train_a1c71cc355c7e9988ab2"),
            (1, 2, "cake_rmsnorm_det_train_ed36a8d998abba41f227"),
            (2, 171, "cake_rmsnorm_det_train_ed36a8d998abba41f227"),
            (171, 16384, "cake_rmsnorm_det_train_cc26e316ae40d683a118"),
            (16384, 32768, "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9"),
            (32768, 55296, "cake_rmsnorm_det_train_302f64db6abe79e5259e"),
            (55296, None, "cake_rmsnorm_det_train_302f64db6abe79e5259e"),
        ),
        "bwd_w32_h2048": (
            (0, 1, "cake_rmsnorm_det_train_ab6b18375f52dcec73f7"),
            (1, 2, "cake_rmsnorm_det_train_978741b43372951cb5b9"),
            (2, 171, "cake_rmsnorm_det_train_978741b43372951cb5b9"),
            (171, 16384, "cake_rmsnorm_det_train_c83c314c6dcf28ad7369"),
            (16384, 32768, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (32768, 55296, "cake_rmsnorm_det_train_e98cb156abfe1666c76f"),
            (55296, None, "cake_rmsnorm_det_train_e98cb156abfe1666c76f"),
        ),
        "bwd_h256": (
            (0, 1680, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (1680, 2048, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (2048, 262144, "cake_rmsnorm_det_train_6bdbfe619b8270ff9a21"),
            (262144, 720896, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (720896, 1572864, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (1572864, None, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
        ),
        "bwd_w32_h256": (
            (0, 1680, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (1680, 2048, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (2048, 262144, "cake_rmsnorm_det_train_6554d4eb4362141cb058"),
            (262144, 720896, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (720896, 1572864, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (1572864, None, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
        ),
        "bwd_h4096": (
            (0, 512, "cake_rmsnorm_det_train_5efbf78a48136634a5b3"),
            (512, None, "cake_rmsnorm_det_train_5572b2671862fe670ef4"),
        ),
        "bwd_w32_h4096": (
            (0, 1, "cake_rmsnorm_det_train_27615d0e36c9f2fe0856"),
            (1, 512, "cake_rmsnorm_det_train_ff7de8d3e4dcd036ea9a"),
            (512, None, "cake_rmsnorm_det_train_47a01388444960da7aae"),
        ),
        "bwd_h512": (
            (0, 1, "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c"),
            (1, 4, "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb"),
            (4, 227, "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb"),
            (227, 1024, "cake_rmsnorm_det_train_09601d42628de24f6f87"),
            (1024, 4096, "cake_rmsnorm_det_train_8e7f4d811e55a351a22e"),
            (4096, 32768, "cake_rmsnorm_det_train_274b0055aace183f2367"),
            (32768, None, "cake_rmsnorm_det_train_cfaae3cec98f8501832b"),
        ),
        "bwd_w32_h512": (
            (0, 1, "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a"),
            (1, 4, "cake_rmsnorm_det_train_2c60470d16e0a7fb6089"),
            (4, 227, "cake_rmsnorm_det_train_2c60470d16e0a7fb6089"),
            (227, 1024, "cake_rmsnorm_det_train_9f07cea1d13b923ed91e"),
            (1024, 4096, "cake_rmsnorm_det_train_dfcbd32cdb8108643fba"),
            (4096, 32768, "cake_rmsnorm_det_train_56ee8e66479625fd078a"),
            (32768, None, "cake_rmsnorm_det_train_a546f410ec8cf03ef20d"),
        ),
        "bwd_h5120": (
            (0, 64, "cake_rmsnorm_det_train_dd3a35731e2e1665dd71"),
            (64, 36544, "cake_rmsnorm_det_train_bfde9b88e800fc34722c"),
            (36544, None, "cake_rmsnorm_det_train_2fc4932fbde0bc848b32"),
        ),
        "bwd_w32_h5120": (
            (0, 64, "cake_rmsnorm_det_train_747d86f34cb58479c7ca"),
            (64, 36544, "cake_rmsnorm_det_train_f9b2d42c4fc7a5a1aa48"),
            (36544, None, "cake_rmsnorm_det_train_79811a76205d7d7dd197"),
        ),
        "bwd_h6144": (
            (0, 256, "cake_rmsnorm_det_train_f2271417b1a596d0540f"),
            (256, 4096, "cake_rmsnorm_det_train_61421bc72e57d14d8a32"),
            (4096, 48640, "cake_rmsnorm_det_train_322574af704b51aef0bb"),
            (48640, None, "cake_rmsnorm_det_train_322574af704b51aef0bb"),
        ),
        "bwd_w32_h6144": (
            (0, 1, "cake_rmsnorm_det_train_b3e6531e46667901ab91"),
            (1, 256, "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5"),
            (256, 4096, "cake_rmsnorm_det_train_41b80b6af97974feb181"),
            (4096, 48640, "cake_rmsnorm_det_train_9a8c0396edd85d4d545e"),
            (48640, None, "cake_rmsnorm_det_train_9a8c0396edd85d4d545e"),
        ),
        "bwd_residual_h6144": (
            (0, 1216, "cake_rmsnorm_det_train_864e84f3d874f850c091"),
            (1216, 48640, "cake_rmsnorm_det_train_04ebec87cbee3f0ce093"),
            (48640, None, "cake_rmsnorm_det_train_04ebec87cbee3f0ce093"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 1216, "cake_rmsnorm_det_train_71d36617003ab7742a77"),
            (1216, 48640, "cake_rmsnorm_det_train_47890c9ae28e3e4b6759"),
            (48640, None, "cake_rmsnorm_det_train_47890c9ae28e3e4b6759"),
        ),
        "fwd_h1024": (
            (0, 2048, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (2048, 4096, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
            (4096, 32768, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (32768, None, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
        ),
        "fwd_w32_h1024": (
            (0, 2, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (2, 4, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (4, 32, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (32, 64, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (64, 512, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (512, 1024, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (1024, 32768, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (32768, None, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
        ),
        "fwd_h128": (
            (0, 16384, "cake_rmsnorm_det_train_3808022dd402387eed16"),
            (16384, None, "cake_rmsnorm_det_train_628267fc724c2c8f0e07"),
        ),
        "fwd_w32_h128": (
            (0, 16384, "cake_rmsnorm_det_train_513f4ffbfefd9d019863"),
            (16384, None, "cake_rmsnorm_det_train_1ecf9c260e418bc70046"),
        ),
        "fwd_h1280": (
            (0, 256, "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5"),
            (256, 1024, "cake_rmsnorm_det_train_03a74dfe62df95c2a752"),
            (1024, 8192, "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5"),
            (8192, 24576, "cake_rmsnorm_det_train_39b9b41b16769252153d"),
            (24576, None, "cake_rmsnorm_det_train_007e82b24390c88156ac"),
        ),
        "fwd_w32_h1280": (
            (0, 512, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (512, 1024, "cake_rmsnorm_det_train_c43709cadf31ed04033c"),
            (1024, 2048, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (2048, 8192, "cake_rmsnorm_det_train_99b209a37f0203af57ce"),
            (8192, 24576, "cake_rmsnorm_det_train_62e8e2756e015c39bf5c"),
            (24576, 32768, "cake_rmsnorm_det_train_58c2fce1ea0b40758e91"),
            (32768, None, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
        ),
        "fwd_h1536": (
            (0, 1, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (1, 256, "cake_rmsnorm_det_train_d52385477d517c951e87"),
            (256, 1536, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_det_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1, "cake_rmsnorm_det_train_21d3416baa877808371e"),
            (1, 256, "cake_rmsnorm_det_train_8e49f86a65049578dee1"),
            (256, 1536, "cake_rmsnorm_det_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_det_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_c5a8099e177e463daab9"),
            (1024, 4096, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_det_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_01f0ebf3db9fa2c0884e"),
            (1024, 4096, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 24, "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8"),
            (24, 8192, "cake_rmsnorm_det_train_0489c784c13bb25499f6"),
            (8192, 20480, "cake_rmsnorm_det_train_bbb05314930d192dfe7b"),
            (20480, 40960, "cake_rmsnorm_det_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_det_train_466331ff3cd2d6c62738"),
            (114688, 196608, "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0"),
            (196608, 327680, "cake_rmsnorm_det_train_ed9483d8c356affee55d"),
            (327680, 368640, "cake_rmsnorm_det_train_37f663183cbff23c2893"),
            (368640, None, "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 24, "cake_rmsnorm_det_train_f8087de00e0409186d93"),
            (24, 8192, "cake_rmsnorm_det_train_f2e721db4c30900f12fe"),
            (8192, 20480, "cake_rmsnorm_det_train_a8fffd93682c5e382dc5"),
            (20480, 40960, "cake_rmsnorm_det_train_77456435c9fbd1d18fcf"),
            (40960, 81920, "cake_rmsnorm_det_train_42d82d26e97c36651dae"),
            (81920, 114688, "cake_rmsnorm_det_train_32bebe5a7782311d60a0"),
            (114688, 196608, "cake_rmsnorm_det_train_42d82d26e97c36651dae"),
            (196608, 327680, "cake_rmsnorm_det_train_d0a656f053396a58c49a"),
            (327680, 368640, "cake_rmsnorm_det_train_18e3c0a5204006b3a56f"),
            (368640, None, "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_det_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
            (1024, 3072, "cake_rmsnorm_det_train_d447b7f6398b60fe5daa"),
            (3072, 6144, "cake_rmsnorm_det_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_det_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
            (1024, 3072, "cake_rmsnorm_det_train_bdc524a233d2bc6e9571"),
            (3072, 6144, "cake_rmsnorm_det_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 4, "cake_rmsnorm_det_train_8757f6da86abf54363d5"),
            (4, 1025, "cake_rmsnorm_det_train_def532441f08b90a9903"),
            (1025, 4096, "cake_rmsnorm_det_train_7673978dc698790054d2"),
            (4096, 16384, "cake_rmsnorm_det_train_3d20ab4a432bddf7349a"),
            (16384, None, "cake_rmsnorm_det_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 4, "cake_rmsnorm_det_train_6769e393999bc2ba1926"),
            (4, 1025, "cake_rmsnorm_det_train_20ebc686cc4099b9d026"),
            (1025, 4096, "cake_rmsnorm_det_train_e2eabcb3b0915d916a13"),
            (4096, 16384, "cake_rmsnorm_det_train_7da36ff557bb2bac8d89"),
            (16384, None, "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 64, "cake_rmsnorm_det_train_6e12e07353014329ebad"),
            (64, 640, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_det_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 64, "cake_rmsnorm_det_train_c1819da87ab45ebb8d56"),
            (64, 640, "cake_rmsnorm_det_train_1defffdf2712cae36784"),
            (640, 6144, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
            (6144, None, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_det_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_det_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (6144, None, "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 4096, "cake_rmsnorm_det_train_8b4730338cac98ec272d"),
            (4096, None, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 4096, "cake_rmsnorm_det_train_3e55e44ce21eff7d8931"),
            (4096, None, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
        ),
    },
    "sm_107a": {
        "bwd_h1024": (
            (0, 1, "cake_rmsnorm_det_train_2dc3fb268514e9182c6a"),
            (1, 8, "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac"),
            (8, 1536, "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2"),
            (1536, 2048, "cake_rmsnorm_det_train_0c581a133624ff396ab4"),
            (2048, 4095, "cake_rmsnorm_det_train_5e32dd889d72a75e348b"),
            (4095, None, "cake_rmsnorm_det_train_4572a519afb83afc3743"),
        ),
        "bwd_w32_h1024": (
            (0, 1, "cake_rmsnorm_det_train_153f811937bb0157224b"),
            (1, 8, "cake_rmsnorm_det_train_cfd067433e487c414dbb"),
            (8, 1536, "cake_rmsnorm_det_train_744d74299c2d45867c94"),
            (1536, 2048, "cake_rmsnorm_det_train_561bea88a25c2f245c29"),
            (2048, 4095, "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca"),
            (4095, None, "cake_rmsnorm_det_train_64423f2da36401d9de39"),
        ),
        "bwd_h128": (
            (0, 1024, "cake_rmsnorm_det_train_9346600882e53a589509"),
            (1024, 2047, "cake_rmsnorm_det_train_401a018c68bc082078b2"),
            (2047, 65536, "cake_rmsnorm_det_train_d197bcc34403e94ee2bd"),
            (65536, None, "cake_rmsnorm_det_train_401a018c68bc082078b2"),
        ),
        "bwd_w32_h128": (
            (0, 1024, "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e"),
            (1024, 2047, "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea"),
            (2047, 65536, "cake_rmsnorm_det_train_e23083c7af388f9e2f50"),
            (65536, None, "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea"),
        ),
        "bwd_h1280": (
            (0, 1, "cake_rmsnorm_det_train_8895b0224bb3d596a3a6"),
            (1, 128, "cake_rmsnorm_det_train_a5f2942d61e591f57a76"),
            (128, 512, "cake_rmsnorm_det_train_809235a3965d337a6877"),
            (512, 6143, "cake_rmsnorm_det_train_12bbedff93fcaafa475d"),
            (6143, 32767, "cake_rmsnorm_det_train_3f26352ef0f1d5e47ecd"),
            (32767, None, "cake_rmsnorm_det_train_7ac3bb6464039170699d"),
        ),
        "bwd_w32_h1280": (
            (0, 1, "cake_rmsnorm_det_train_103805d93149c8668264"),
            (1, 128, "cake_rmsnorm_det_train_8d8ef930b7d650c5538a"),
            (128, 512, "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d"),
            (512, 4096, "cake_rmsnorm_det_train_80956d258172dfa76c2a"),
            (4096, 6143, "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4"),
            (6143, 32767, "cake_rmsnorm_det_train_b1a35315ca6fcda95b3c"),
            (32767, None, "cake_rmsnorm_det_train_7a4606b019405eb9a2ef"),
        ),
        "bwd_h1536": (
            (0, 1, "cake_rmsnorm_det_train_c3b1b46a2c6d1f9b3724"),
            (1, 576, "cake_rmsnorm_det_train_1207b05663ed2b4de14b"),
            (576, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 12288, "cake_rmsnorm_det_train_88510c609fd688aab969"),
            (12288, 16384, "cake_rmsnorm_det_train_6a428da2227972ae18ae"),
            (16384, 45056, "cake_rmsnorm_det_train_56d26996239835327501"),
            (45056, None, "cake_rmsnorm_det_train_411e270c0575d26a77f4"),
        ),
        "bwd_w32_h1536": (
            (0, 1, "cake_rmsnorm_det_train_45057834f9e0e017b6ef"),
            (1, 576, "cake_rmsnorm_det_train_137ba5328176fb07b37d"),
            (576, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 12288, "cake_rmsnorm_det_train_6f2c9e90f551f35e99e6"),
            (12288, 16384, "cake_rmsnorm_det_train_e63beae98c236ce26335"),
            (16384, 45056, "cake_rmsnorm_det_train_de8e07d4cb312b06dbc4"),
            (45056, None, "cake_rmsnorm_det_train_64ffe19b041f5bfa34fb"),
        ),
        "bwd_h2048": (
            (0, 1, "cake_rmsnorm_det_train_92def3f97ddcec9165f0"),
            (1, 256, "cake_rmsnorm_det_train_6720b4a159be2b84b325"),
            (256, 1024, "cake_rmsnorm_det_train_845cdcd8145ada631461"),
            (1024, 16384, "cake_rmsnorm_det_train_cc26e316ae40d683a118"),
            (16384, 50688, "cake_rmsnorm_det_train_845cdcd8145ada631461"),
            (50688, None, "cake_rmsnorm_det_train_21ee6fc93139c5e0cd41"),
        ),
        "bwd_w32_h2048": (
            (0, 1, "cake_rmsnorm_det_train_5e3034bbb4032d8b7ad9"),
            (1, 256, "cake_rmsnorm_det_train_23c900134dc889302136"),
            (256, 1024, "cake_rmsnorm_det_train_838a9a4054731f7942af"),
            (1024, 16384, "cake_rmsnorm_det_train_c83c314c6dcf28ad7369"),
            (16384, 50688, "cake_rmsnorm_det_train_838a9a4054731f7942af"),
            (50688, None, "cake_rmsnorm_det_train_3f270477a248188afe2a"),
        ),
        "bwd_h256": (
            (0, 1572864, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (1572864, None, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
        ),
        "bwd_w32_h256": (
            (0, 1572864, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (1572864, None, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
        ),
        "bwd_h4096": (
            (0, 256, "cake_rmsnorm_det_train_c1de7e41da4550cca0bc"),
            (256, 1024, "cake_rmsnorm_det_train_a647fccc3d9fa9134733"),
            (1024, 4096, "cake_rmsnorm_det_train_d41a514f5210e9adc39f"),
            (4096, 24064, "cake_rmsnorm_det_train_ee4a9053bb25b8d49fcf"),
            (24064, 50176, "cake_rmsnorm_det_train_a647fccc3d9fa9134733"),
            (50176, None, "cake_rmsnorm_det_train_e9e21bbe4fa478075868"),
        ),
        "bwd_w32_h4096": (
            (0, 256, "cake_rmsnorm_det_train_30ec8bb68cf4fe0c817b"),
            (256, 1024, "cake_rmsnorm_det_train_dc439f8335327e482025"),
            (1024, 4096, "cake_rmsnorm_det_train_b08633208e60b8ce94fd"),
            (4096, 24064, "cake_rmsnorm_det_train_0f1859b9fe9e47668a1c"),
            (24064, 50176, "cake_rmsnorm_det_train_0ff440daa6f9f304e544"),
            (50176, None, "cake_rmsnorm_det_train_0ff440daa6f9f304e544"),
        ),
        "bwd_h512": (
            (0, 1, "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c"),
            (1, 768, "cake_rmsnorm_det_train_b143f616f21eeeaa021d"),
            (768, 1024, "cake_rmsnorm_det_train_3feed5026e1156d2fba2"),
            (1024, 4096, "cake_rmsnorm_det_train_3feed5026e1156d2fba2"),
            (4096, 16384, "cake_rmsnorm_det_train_dfaf378aa8ad93e68bbd"),
            (16384, None, "cake_rmsnorm_det_train_eb141cc9e0a93c6f0ce3"),
        ),
        "bwd_w32_h512": (
            (0, 1, "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a"),
            (1, 768, "cake_rmsnorm_det_train_740c03baae3da1ceb102"),
            (768, 1024, "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b"),
            (1024, 4096, "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b"),
            (4096, 16384, "cake_rmsnorm_det_train_383f709cd5c4fadd9436"),
            (16384, None, "cake_rmsnorm_det_train_03efa681eb38f59b00c2"),
        ),
        "bwd_h5120": (
            (0, 1, "cake_rmsnorm_det_train_080e0a6c3b18e9bad790"),
            (1, 256, "cake_rmsnorm_det_train_dd3a35731e2e1665dd71"),
            (256, 1280, "cake_rmsnorm_det_train_080e0a6c3b18e9bad790"),
            (1280, 4096, "cake_rmsnorm_det_train_838c077df3a6d35d55b7"),
            (4096, 19968, "cake_rmsnorm_det_train_6816fb80e97380f1bcbc"),
            (19968, None, "cake_rmsnorm_det_train_47da668e7e96652f840b"),
        ),
        "bwd_w32_h5120": (
            (0, 1, "cake_rmsnorm_det_train_139d162a8bb29d1b2d35"),
            (1, 256, "cake_rmsnorm_det_train_747d86f34cb58479c7ca"),
            (256, 1280, "cake_rmsnorm_det_train_139d162a8bb29d1b2d35"),
            (1280, 4096, "cake_rmsnorm_det_train_cd612de56c0985bb4caa"),
            (4096, 19968, "cake_rmsnorm_det_train_468cc4d1b8128187b4b3"),
            (19968, None, "cake_rmsnorm_det_train_de0cf457cd240f9c6422"),
        ),
        "bwd_h6144": (
            (0, 1, "cake_rmsnorm_det_train_0dafdf094615d080fcc9"),
            (1, 256, "cake_rmsnorm_det_train_dffe53d5aae62879b794"),
            (256, 1664, "cake_rmsnorm_det_train_0dafdf094615d080fcc9"),
            (1664, 3072, "cake_rmsnorm_det_train_af220b26c5465e3f8b94"),
            (3072, 19456, "cake_rmsnorm_det_train_5d292172df772adb17ae"),
            (19456, None, "cake_rmsnorm_det_train_9cc51adfe9826056658b"),
        ),
        "bwd_w32_h6144": (
            (0, 1, "cake_rmsnorm_det_train_5cca2d0fa4553f992406"),
            (1, 256, "cake_rmsnorm_det_train_276474ee04bad8d3d764"),
            (256, 1664, "cake_rmsnorm_det_train_5cca2d0fa4553f992406"),
            (1664, 3072, "cake_rmsnorm_det_train_ac531ff9353aad7a17c6"),
            (3072, 19456, "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb"),
            (19456, None, "cake_rmsnorm_det_train_bdfba721a43e0d093668"),
        ),
        "bwd_residual_h6144": (
            (0, 368, "cake_rmsnorm_det_train_ee1be07a89a0482de45e"),
            (368, 19456, "cake_rmsnorm_det_train_36cad444bc992ccc732d"),
            (19456, None, "cake_rmsnorm_det_train_2f1499999535c6983213"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 368, "cake_rmsnorm_det_train_3bbd4f848b4847bff893"),
            (368, 19456, "cake_rmsnorm_det_train_b9d848cdbcfaeea0325c"),
            (19456, None, "cake_rmsnorm_det_train_56d25337feb636c80b9a"),
        ),
        "fwd_h1024": (
            (0, 8, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (8, 16, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
            (16, 256, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (256, 1024, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
            (1024, 4096, "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e"),
            (4096, 8192, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
            (8192, 8447, "cake_rmsnorm_det_train_cffbc5c8de9c874b207c"),
            (8447, 13312, "cake_rmsnorm_det_train_547372db70c0ca0be362"),
            (13312, 32768, "cake_rmsnorm_det_train_9e8a70075f28be558bba"),
            (32768, None, "cake_rmsnorm_det_train_cffbc5c8de9c874b207c"),
        ),
        "fwd_w32_h1024": (
            (0, 8, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (8, 16, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (16, 512, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (512, 1536, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (1536, 4096, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (4096, 8192, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (8192, 12288, "cake_rmsnorm_det_train_f5110367098beea13130"),
            (12288, 16384, "cake_rmsnorm_det_train_364ed18ca31e5b754d4e"),
            (16384, 16895, "cake_rmsnorm_det_train_8bd2d83654da4780c48a"),
            (16895, 24576, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (24576, 32768, "cake_rmsnorm_det_train_d99009bfce4043a5dffe"),
            (32768, None, "cake_rmsnorm_det_train_008fb179743126fb2d35"),
        ),
        "fwd_h128": (
            (0, 32, "cake_rmsnorm_det_train_3808022dd402387eed16"),
            (32, 64, "cake_rmsnorm_det_train_628267fc724c2c8f0e07"),
            (64, 2048, "cake_rmsnorm_det_train_3808022dd402387eed16"),
            (2048, 4096, "cake_rmsnorm_det_train_7221de56ded18727adaf"),
            (4096, 24576, "cake_rmsnorm_det_train_3808022dd402387eed16"),
            (24576, 32768, "cake_rmsnorm_det_train_7221de56ded18727adaf"),
            (32768, None, "cake_rmsnorm_det_train_a05a2262fd5fa6912442"),
        ),
        "fwd_w32_h128": (
            (0, 32, "cake_rmsnorm_det_train_513f4ffbfefd9d019863"),
            (32, 64, "cake_rmsnorm_det_train_1ecf9c260e418bc70046"),
            (64, 2048, "cake_rmsnorm_det_train_513f4ffbfefd9d019863"),
            (2048, 4096, "cake_rmsnorm_det_train_04ea6484fdf4a9b14ee0"),
            (4096, 16384, "cake_rmsnorm_det_train_513f4ffbfefd9d019863"),
            (16384, 32768, "cake_rmsnorm_det_train_1ecf9c260e418bc70046"),
            (32768, None, "cake_rmsnorm_det_train_67e0dfc12fbae0354e51"),
        ),
        "fwd_h1280": (
            (0, 1, "cake_rmsnorm_det_train_03a74dfe62df95c2a752"),
            (1, 4, "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5"),
            (4, 8, "cake_rmsnorm_det_train_03a74dfe62df95c2a752"),
            (8, 16, "cake_rmsnorm_det_train_007e82b24390c88156ac"),
            (16, 128, "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5"),
            (128, 512, "cake_rmsnorm_det_train_03a74dfe62df95c2a752"),
            (512, 1024, "cake_rmsnorm_det_train_007e82b24390c88156ac"),
            (1024, 1536, "cake_rmsnorm_det_train_03b5516039c533b453b9"),
            (1536, 2048, "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5"),
            (2048, 4096, "cake_rmsnorm_det_train_03a74dfe62df95c2a752"),
            (4096, 12288, "cake_rmsnorm_det_train_007e82b24390c88156ac"),
            (12288, 16384, "cake_rmsnorm_det_train_03b5516039c533b453b9"),
            (16384, None, "cake_rmsnorm_det_train_007e82b24390c88156ac"),
        ),
        "fwd_w32_h1280": (
            (0, 4, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (4, 8, "cake_rmsnorm_det_train_c43709cadf31ed04033c"),
            (8, 128, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (128, 512, "cake_rmsnorm_det_train_c43709cadf31ed04033c"),
            (512, 1024, "cake_rmsnorm_det_train_58c2fce1ea0b40758e91"),
            (1024, 1536, "cake_rmsnorm_det_train_62e8e2756e015c39bf5c"),
            (1536, 2048, "cake_rmsnorm_det_train_6258b6dac6840cc6bd58"),
            (2048, 8192, "cake_rmsnorm_det_train_62e8e2756e015c39bf5c"),
            (8192, 8703, "cake_rmsnorm_det_train_99b209a37f0203af57ce"),
            (8703, 9984, "cake_rmsnorm_det_train_51ca3c875efd4b1a5746"),
            (9984, 16384, "cake_rmsnorm_det_train_99b209a37f0203af57ce"),
            (16384, None, "cake_rmsnorm_det_train_58c2fce1ea0b40758e91"),
        ),
        "fwd_h1536": (
            (0, 2048, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (2048, 12288, "cake_rmsnorm_det_train_4cdc4c495b3f59ff5069"),
            (12288, None, "cake_rmsnorm_det_train_682675bda578f5b7a312"),
        ),
        "fwd_w32_h1536": (
            (0, 2048, "cake_rmsnorm_det_train_21d3416baa877808371e"),
            (2048, 12288, "cake_rmsnorm_det_train_92cf7b87f1ebffb1b12b"),
            (12288, None, "cake_rmsnorm_det_train_a30e5a3b69b7c183992f"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (1024, 8192, "cake_rmsnorm_det_train_a4da85136fba4588239f"),
            (8192, 12288, "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d"),
            (12288, 16384, "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d"),
            (16384, None, "cake_rmsnorm_det_train_bde56bd5a68c493f07a2"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (1024, 8192, "cake_rmsnorm_det_train_a72c6ece1900f748eaa2"),
            (8192, 12288, "cake_rmsnorm_det_train_595f506004038a65cc63"),
            (12288, 16384, "cake_rmsnorm_det_train_40210950f2e7ee48bc3e"),
            (16384, None, "cake_rmsnorm_det_train_40210950f2e7ee48bc3e"),
        ),
        "fwd_h256": (
            (0, 6144, "cake_rmsnorm_det_train_c5ca409f44e169148301"),
            (6144, 32768, "cake_rmsnorm_det_train_bbb05314930d192dfe7b"),
            (32768, 98304, "cake_rmsnorm_det_train_466331ff3cd2d6c62738"),
            (98304, None, "cake_rmsnorm_det_train_37f663183cbff23c2893"),
        ),
        "fwd_w32_h256": (
            (0, 6144, "cake_rmsnorm_det_train_8f2756937eaac6b02502"),
            (6144, 32768, "cake_rmsnorm_det_train_a8fffd93682c5e382dc5"),
            (32768, 98304, "cake_rmsnorm_det_train_32bebe5a7782311d60a0"),
            (98304, None, "cake_rmsnorm_det_train_18e3c0a5204006b3a56f"),
        ),
        "fwd_h4096": (
            (0, 512, "cake_rmsnorm_det_train_8392d1a008d58a006bba"),
            (512, 1024, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
            (1024, 4096, "cake_rmsnorm_det_train_d447b7f6398b60fe5daa"),
            (4096, 20480, "cake_rmsnorm_det_train_bf933e1660cab8e5787b"),
            (20480, 49152, "cake_rmsnorm_det_train_bf933e1660cab8e5787b"),
            (49152, None, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 512, "cake_rmsnorm_det_train_4371270e4db4b985998f"),
            (512, 1024, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
            (1024, 4096, "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794"),
            (4096, 20480, "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794"),
            (20480, 49152, "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794"),
            (49152, None, "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794"),
        ),
        "fwd_h512": (
            (0, 256, "cake_rmsnorm_det_train_7673978dc698790054d2"),
            (256, 6144, "cake_rmsnorm_det_train_2a42e5c02b240f8709b1"),
            (6144, None, "cake_rmsnorm_det_train_60f47c5d00570ecf7740"),
        ),
        "fwd_w32_h512": (
            (0, 256, "cake_rmsnorm_det_train_e2eabcb3b0915d916a13"),
            (256, 6144, "cake_rmsnorm_det_train_79fecc7016e44e940733"),
            (6144, None, "cake_rmsnorm_det_train_61accbcc02891615f286"),
        ),
        "fwd_h5120": (
            (0, 512, "cake_rmsnorm_det_train_fd1097c45c1e328871fd"),
            (512, 1152, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
            (1152, 6144, "cake_rmsnorm_det_train_b5e173f5ecc736d91109"),
            (6144, 20480, "cake_rmsnorm_det_train_044d4a2aa0553c16db06"),
            (20480, 49152, "cake_rmsnorm_det_train_044d4a2aa0553c16db06"),
            (49152, None, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 512, "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e"),
            (512, 1152, "cake_rmsnorm_det_train_1defffdf2712cae36784"),
            (1152, 6144, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
            (6144, 20480, "cake_rmsnorm_det_train_b49de9b47372fa4e1051"),
            (20480, 49152, "cake_rmsnorm_det_train_b49de9b47372fa4e1051"),
            (49152, None, "cake_rmsnorm_det_train_b49de9b47372fa4e1051"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_det_train_bad4db9153cc903b7ace"),
            (1024, 4096, "cake_rmsnorm_det_train_cf1d05546951bdbd9f5d"),
            (4096, None, "cake_rmsnorm_det_train_e7f4221a7c26b9dfdd23"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (1024, 4096, "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d"),
            (4096, None, "cake_rmsnorm_det_train_fdfa500c83d95df0f5c1"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
            (256, 1024, "cake_rmsnorm_det_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
            (256, 1024, "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
        ),
    },
}

# Per architecture and route: the upper edge of every window but the last and the measured
# class of the outputs across it: "same" (bitwise identical on the common rows), "ulp" (y
# moves by at most one BF16 ulp, rstd by at most two FP32 ulps, three on R200 (216 SMs) at
# the residual route's edge 1024), "dx" (the backward kernel switch), "unmeasured".
EDGES: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {
    "sm_100a": {
        "bwd_h1024": (
            (1, "unmeasured"),
            (64, "unmeasured"),
            (128, "unmeasured"),
            (1024, "unmeasured"),
            (8192, "unmeasured"),
        ),
        "bwd_w32_h1024": (
            (1, "unmeasured"),
            (1024, "unmeasured"),
            (8192, "unmeasured"),
        ),
        "bwd_h128": ((1024, "unmeasured"), (2048, "unmeasured"), (11201, "unmeasured")),
        "bwd_w32_h128": (
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (11201, "unmeasured"),
        ),
        "bwd_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (4096, "unmeasured"),
        ),
        "bwd_w32_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (4096, "unmeasured"),
        ),
        "bwd_h1536": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "same"),
            (1184, "unmeasured"),
            (161792, "unmeasured"),
        ),
        "bwd_w32_h1536": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "same"),
            (1184, "unmeasured"),
            (161792, "unmeasured"),
        ),
        "bwd_h2048": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "unmeasured"),
            (2047, "unmeasured"),
            (22528, "unmeasured"),
            (58880, "unmeasured"),
        ),
        "bwd_w32_h2048": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "unmeasured"),
            (16384, "same"),
            (58880, "unmeasured"),
        ),
        "bwd_h256": (
            (87, "unmeasured"),
            (2048, "same"),
            (5120, "same"),
            (12288, "unmeasured"),
            (262144, "unmeasured"),
            (1572864, "same"),
        ),
        "bwd_w32_h256": (
            (87, "unmeasured"),
            (2048, "same"),
            (5120, "same"),
            (12288, "unmeasured"),
            (262144, "unmeasured"),
            (1572864, "same"),
        ),
        "bwd_h4096": (
            (203, "unmeasured"),
            (256, "same"),
            (1024, "unmeasured"),
            (22528, "unmeasured"),
            (32768, "unmeasured"),
            (44032, "same"),
        ),
        "bwd_w32_h4096": (
            (203, "unmeasured"),
            (256, "same"),
            (1024, "unmeasured"),
            (22528, "unmeasured"),
            (32768, "unmeasured"),
            (44032, "same"),
        ),
        "bwd_h512": (
            (1, "unmeasured"),
            (227, "unmeasured"),
            (1024, "ulp"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_w32_h512": (
            (1, "unmeasured"),
            (227, "unmeasured"),
            (1024, "ulp"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_h5120": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (36864, "unmeasured"),
            (50688, "same"),
        ),
        "bwd_w32_h5120": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (36864, "unmeasured"),
            (50688, "unmeasured"),
        ),
        "bwd_h6144": ((384, "unmeasured"), (4096, "unmeasured"), (42496, "same")),
        "bwd_w32_h6144": ((384, "unmeasured"), (4096, "same"), (42496, "same")),
        "bwd_residual_h6144": ((1152, "unmeasured"), (42496, "unmeasured")),
        "bwd_w32_residual_h6144": ((1152, "unmeasured"), (42496, "unmeasured")),
        "fwd_h1024": (
            (2, "unmeasured"),
            (4, "unmeasured"),
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (4096, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_w32_h1024": (
            (2, "unmeasured"),
            (4, "unmeasured"),
            (32, "unmeasured"),
            (64, "unmeasured"),
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_h128": ((2, "unmeasured"), (4, "unmeasured"), (16384, "unmeasured")),
        "fwd_w32_h128": (
            (2, "unmeasured"),
            (4, "unmeasured"),
            (64, "unmeasured"),
            (128, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "fwd_h1280": (
            (16, "unmeasured"),
            (32, "unmeasured"),
            (256, "unmeasured"),
            (512, "unmeasured"),
            (8192, "unmeasured"),
            (24576, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_w32_h1280": (
            (1, "unmeasured"),
            (4, "unmeasured"),
            (8, "unmeasured"),
            (16, "unmeasured"),
            (256, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_w32_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_h2048": (
            (1, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "ulp"),
            (4096, "ulp"),
        ),
        "fwd_w32_h2048": (
            (1, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "ulp"),
            (4096, "ulp"),
        ),
        "fwd_h256": (
            (1024, "unmeasured"),
            (8192, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
        ),
        "fwd_w32_h256": (
            (1024, "unmeasured"),
            (8192, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
        ),
        "fwd_h4096": ((640, "ulp"), (1024, "same"), (6144, "same")),
        "fwd_w32_h4096": ((640, "ulp"), (1024, "same"), (6144, "same")),
        "fwd_h512": ((1024, "ulp"), (4096, "same"), (24576, "same")),
        "fwd_w32_h512": ((1024, "ulp"), (4096, "same"), (24576, "same")),
        "fwd_h5120": ((64, "unmeasured"), (640, "same"), (6144, "same")),
        "fwd_w32_h5120": (
            (64, "unmeasured"),
            (640, "same"),
            (6144, "same"),
            (16384, "same"),
        ),
        "fwd_h6144": ((768, "same"), (1024, "same")),
        "fwd_w32_h6144": ((768, "same"), (1024, "same"), (6144, "same")),
        "fwd_residual_h6144": (
            (256, "unmeasured"),
            (1024, "ulp"),
            (4096, "unmeasured"),
        ),
        "fwd_w32_residual_h6144": (
            (256, "unmeasured"),
            (1024, "ulp"),
            (4096, "unmeasured"),
        ),
    },
    "sm_103a": {
        "bwd_h1024": (
            (1, "unmeasured"),
            (64, "unmeasured"),
            (128, "unmeasured"),
            (1024, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (4096, "unmeasured"),
            (8192, "unmeasured"),
            (12288, "unmeasured"),
        ),
        "bwd_w32_h1024": (
            (1, "unmeasured"),
            (1024, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (12288, "unmeasured"),
        ),
        "bwd_h128": ((1024, "unmeasured"), (2048, "unmeasured"), (12288, "unmeasured")),
        "bwd_w32_h128": (
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (12288, "unmeasured"),
        ),
        "bwd_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (2048, "unmeasured"),
        ),
        "bwd_w32_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (4096, "unmeasured"),
        ),
        "bwd_h1536": (
            (1, "unmeasured"),
            (31, "unmeasured"),
            (214, "unmeasured"),
            (1024, "same"),
            (1216, "unmeasured"),
            (214016, "unmeasured"),
        ),
        "bwd_w32_h1536": (
            (1, "unmeasured"),
            (31, "unmeasured"),
            (214, "unmeasured"),
            (1024, "same"),
            (1216, "unmeasured"),
            (214016, "unmeasured"),
        ),
        "bwd_h2048": (
            (1, "unmeasured"),
            (2, "unmeasured"),
            (171, "unmeasured"),
            (16384, "unmeasured"),
            (32768, "same"),
            (55296, "same"),
        ),
        "bwd_w32_h2048": (
            (1, "unmeasured"),
            (2, "unmeasured"),
            (171, "unmeasured"),
            (16384, "unmeasured"),
            (32768, "same"),
            (55296, "same"),
        ),
        "bwd_h256": (
            (1680, "unmeasured"),
            (2048, "unmeasured"),
            (262144, "unmeasured"),
            (720896, "same"),
            (1572864, "same"),
        ),
        "bwd_w32_h256": (
            (1680, "unmeasured"),
            (2048, "unmeasured"),
            (262144, "unmeasured"),
            (720896, "same"),
            (1572864, "same"),
        ),
        "bwd_h4096": ((512, "unmeasured"),),
        "bwd_w32_h4096": ((1, "unmeasured"), (512, "unmeasured")),
        "bwd_h512": (
            (1, "unmeasured"),
            (4, "unmeasured"),
            (227, "unmeasured"),
            (1024, "unmeasured"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_w32_h512": (
            (1, "unmeasured"),
            (4, "unmeasured"),
            (227, "unmeasured"),
            (1024, "unmeasured"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_h5120": ((64, "unmeasured"), (36544, "unmeasured")),
        "bwd_w32_h5120": ((64, "unmeasured"), (36544, "unmeasured")),
        "bwd_h6144": ((256, "unmeasured"), (4096, "unmeasured"), (48640, "same")),
        "bwd_w32_h6144": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (4096, "same"),
            (48640, "same"),
        ),
        "bwd_residual_h6144": ((1216, "unmeasured"), (48640, "unmeasured")),
        "bwd_w32_residual_h6144": ((1216, "unmeasured"), (48640, "unmeasured")),
        "fwd_h1024": (
            (2048, "unmeasured"),
            (4096, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_w32_h1024": (
            (2, "unmeasured"),
            (4, "unmeasured"),
            (32, "unmeasured"),
            (64, "unmeasured"),
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_h128": ((16384, "unmeasured"),),
        "fwd_w32_h128": ((16384, "unmeasured"),),
        "fwd_h1280": (
            (256, "unmeasured"),
            (1024, "unmeasured"),
            (8192, "unmeasured"),
            (24576, "unmeasured"),
        ),
        "fwd_w32_h1280": (
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (8192, "unmeasured"),
            (24576, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_h1536": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1536, "ulp"),
            (3584, "same"),
            (6144, "same"),
        ),
        "fwd_w32_h1536": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1536, "ulp"),
            (3584, "same"),
            (6144, "same"),
        ),
        "fwd_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_w32_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_h256": (
            (24, "unmeasured"),
            (8192, "same"),
            (20480, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
            (368640, "same"),
        ),
        "fwd_w32_h256": (
            (24, "unmeasured"),
            (8192, "same"),
            (20480, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
            (368640, "same"),
        ),
        "fwd_h4096": ((640, "ulp"), (1024, "same"), (3072, "same"), (6144, "same")),
        "fwd_w32_h4096": ((640, "ulp"), (1024, "same"), (3072, "same"), (6144, "same")),
        "fwd_h512": (
            (4, "unmeasured"),
            (1025, "unmeasured"),
            (4096, "same"),
            (16384, "same"),
        ),
        "fwd_w32_h512": (
            (4, "unmeasured"),
            (1025, "unmeasured"),
            (4096, "same"),
            (16384, "same"),
        ),
        "fwd_h5120": ((64, "unmeasured"), (640, "same"), (6144, "same")),
        "fwd_w32_h5120": ((64, "unmeasured"), (640, "same"), (6144, "same")),
        "fwd_h6144": ((1024, "same"),),
        "fwd_w32_h6144": ((1024, "same"), (6144, "same")),
        "fwd_residual_h6144": ((4096, "unmeasured"),),
        "fwd_w32_residual_h6144": ((4096, "unmeasured"),),
    },
    "sm_107a": {
        "bwd_h1024": (
            (1, "unmeasured"),
            (8, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (4095, "unmeasured"),
        ),
        "bwd_w32_h1024": (
            (1, "unmeasured"),
            (8, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (4095, "unmeasured"),
        ),
        "bwd_h128": ((1024, "unmeasured"), (2047, "unmeasured"), (65536, "unmeasured")),
        "bwd_w32_h128": (
            (1024, "unmeasured"),
            (2047, "unmeasured"),
            (65536, "unmeasured"),
        ),
        "bwd_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (6143, "unmeasured"),
            (32767, "unmeasured"),
        ),
        "bwd_w32_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (4096, "unmeasured"),
            (6143, "unmeasured"),
            (32767, "unmeasured"),
        ),
        "bwd_h1536": (
            (1, "unmeasured"),
            (576, "unmeasured"),
            (1024, "same"),
            (12288, "same"),
            (16384, "same"),
            (45056, "same"),
        ),
        "bwd_w32_h1536": (
            (1, "unmeasured"),
            (576, "unmeasured"),
            (1024, "same"),
            (12288, "same"),
            (16384, "same"),
            (45056, "same"),
        ),
        "bwd_h2048": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1024, "ulp"),
            (16384, "same"),
            (50688, "same"),
        ),
        "bwd_w32_h2048": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1024, "ulp"),
            (16384, "same"),
            (50688, "same"),
        ),
        "bwd_h256": ((1572864, "same"),),
        "bwd_w32_h256": ((1572864, "same"),),
        "bwd_h4096": (
            (256, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (24064, "unmeasured"),
            (50176, "unmeasured"),
        ),
        "bwd_w32_h4096": (
            (256, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (24064, "same"),
            (50176, "same"),
        ),
        "bwd_h512": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (16384, "unmeasured"),
        ),
        "bwd_w32_h512": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (16384, "unmeasured"),
        ),
        "bwd_h5120": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1280, "unmeasured"),
            (4096, "unmeasured"),
            (19968, "unmeasured"),
        ),
        "bwd_w32_h5120": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1280, "unmeasured"),
            (4096, "unmeasured"),
            (19968, "unmeasured"),
        ),
        "bwd_h6144": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1664, "dx"),
            (3072, "dx"),
            (19456, "same"),
        ),
        "bwd_w32_h6144": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1664, "dx"),
            (3072, "dx"),
            (19456, "same"),
        ),
        "bwd_residual_h6144": ((368, "unmeasured"), (19456, "unmeasured")),
        "bwd_w32_residual_h6144": ((368, "unmeasured"), (19456, "unmeasured")),
        "fwd_h1024": (
            (8, "unmeasured"),
            (16, "unmeasured"),
            (256, "unmeasured"),
            (1024, "unmeasured"),
            (4096, "unmeasured"),
            (8192, "unmeasured"),
            (8447, "unmeasured"),
            (13312, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_w32_h1024": (
            (8, "unmeasured"),
            (16, "unmeasured"),
            (512, "unmeasured"),
            (1536, "unmeasured"),
            (4096, "unmeasured"),
            (8192, "unmeasured"),
            (12288, "unmeasured"),
            (16384, "unmeasured"),
            (16895, "unmeasured"),
            (24576, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_h128": (
            (32, "unmeasured"),
            (64, "unmeasured"),
            (2048, "unmeasured"),
            (4096, "unmeasured"),
            (24576, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_w32_h128": (
            (32, "unmeasured"),
            (64, "unmeasured"),
            (2048, "unmeasured"),
            (4096, "unmeasured"),
            (16384, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "fwd_h1280": (
            (1, "unmeasured"),
            (4, "unmeasured"),
            (8, "unmeasured"),
            (16, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (4096, "unmeasured"),
            (12288, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "fwd_w32_h1280": (
            (4, "unmeasured"),
            (8, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (8192, "unmeasured"),
            (8703, "unmeasured"),
            (9984, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "fwd_h1536": ((2048, "ulp"), (12288, "same")),
        "fwd_w32_h1536": ((2048, "ulp"), (12288, "same")),
        "fwd_h2048": ((1024, "ulp"), (8192, "same"), (12288, "same"), (16384, "same")),
        "fwd_w32_h2048": (
            (1024, "ulp"),
            (8192, "same"),
            (12288, "same"),
            (16384, "same"),
        ),
        "fwd_h256": ((6144, "same"), (32768, "same"), (98304, "same")),
        "fwd_w32_h256": ((6144, "same"), (32768, "same"), (98304, "same")),
        "fwd_h4096": (
            (512, "ulp"),
            (1024, "same"),
            (4096, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_w32_h4096": (
            (512, "ulp"),
            (1024, "same"),
            (4096, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_h512": ((256, "same"), (6144, "same")),
        "fwd_w32_h512": ((256, "same"), (6144, "same")),
        "fwd_h5120": (
            (512, "same"),
            (1152, "same"),
            (6144, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_w32_h5120": (
            (512, "same"),
            (1152, "same"),
            (6144, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_h6144": ((1024, "same"), (4096, "same")),
        "fwd_w32_h6144": ((1024, "same"), (4096, "same")),
        "fwd_residual_h6144": ((256, "ulp"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "ulp"), (1024, "ulp")),
    },
}

EDGE_CLASSES = ["same", "ulp", "dx", "unmeasured"]

# The backward accuracy guard (see NOTES): per architecture the smallest SM count the accuracy
# edges were measured on, and per route the edge above which it runs compensated programs.  On
# a device with fewer SMs a backward call of such a route selects its programs as if ``T`` were
# the edge plus one (its launch keeps ``T``).
# The guard holds no program compensated, so no route has an edge.
ACCURACY_MIN_SMS: dict[str, int] = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 212}
ACCURACY_EDGES: dict[str, dict[str, int]] = {}

NOTES = """
Each route -- one architecture, hidden size and direction -- ships several programs of
one kernel and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  Within one window a row's ``y``, ``rstd`` and ``dx`` do
not depend on ``T``: two calls whose token counts fall in the same forward window (and,
for the backward, the same backward window) return bitwise-identical results on their
common rows.  Across a forward edge between programs of different per-row thread groups
the sum of squares is reduced in another order.  On the rows both programs compute,
``rstd`` then moves by at most two FP32 ulps (three on SM107 at the residual route's
edge 1024) on 10.8-18.2 % of the rows (24.0-31.6 % on the residual route); ``y`` moves
by one BF16 ulp, only on rows whose ``rstd`` moved, on at most 2.34 % of the rows
(4.30 % on the residual route); ``dx`` follows on those rows and the residual output
``h_new`` does not change.  Across every other measured forward edge, every measured
edge of H = 256 and H = 5120 among them, the outputs are bitwise identical.  The
programs of the smallest token counts that form ``rstd`` with the approximate reciprocal
square root (``rsqrt.approx``) give other ``rstd`` and ``y`` bits than the programs
beside them, so the outputs can change bits at their edges; ``EDGES`` marks those edges
as not yet measured.  Measured over three random inputs per edge, with BF16 and FP32
weights: on B200 (sm_100a) at most two ulps on up to 18.0 % of the rows (27.3 % on the
residual route), ``y`` on up to 2.03 % (2.83 %); on GB300 (sm_103a) at most two ulps on
up to 17.3 % (27.8 %), ``y`` on up to 2.03 % (2.34 %); on R200 (sm_107a, 212- and 216-SM
boards) at most three ulps on up to 18.2 % (31.6 %), ``y`` on up to 2.34 % (4.30 %).
Across a measured backward-only edge ``dx`` is bitwise identical but at the sm_107a
H = 6144 edges at 1664 and 3072 tokens, where the programs on the two sides (the
column-task and the one-row-per-stage program at 1664, the one-row-per-stage and the
bulk-copy program at 3072) reduce a row in different orders: ``dx`` differs on 23 of
the 1664 and on 69 of the 3072 common rows of the input measured, by at most two and
six BF16 bit-pattern steps, while ``y`` and ``rstd`` are identical.  The H = 6144
route, which switched between the base backward kernel and the bulk-copy one with
``dx`` differing on about 1-2 % of the rows, now runs column-task programs up to 384
tokens on sm_100a, 256 on sm_103a and 1664 on sm_107a, one-row-per-stage programs over
(384, 4096] on sm_100a and (256, 4096] on sm_103a (BF16 weight; the FP32-weight route
runs the bulk-copy kernel there) and over (1664, 3072] on sm_107a, and the bulk-copy
kernel above.  The one-row programs at ``T`` = 1 (H = 512, 1024, 1280, 1536 and
2048) fold the row's dot product in their own fixed order, so ``dx`` may change bits at
their edge; ``EDGES`` marks the edges not yet measured, every edge of the H = 1280, 1024
and 128 routes among them. ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` it is bitwise reproducible on devices with the same SM
count.  The backward splits the rows over ``ctas_per_sm`` x SM count CTAs, so devices of
one architecture with different SM counts (B200 with 148 and GB200 with 152 SMs, R200
boards with 212 and 216 SMs) cut different chunk tables and can give different ``dw``
bits.  In the windows of its smallest token counts a backward route other than H = 256
(on sm_100a H = 256 too, to 87 rows; the H = 128 routes at every ``T``)
runs column-task programs instead: one CTA sums each column of ``dw`` over every row in
a fixed order in FP32, so there ``dw`` does not depend on the SM count.  Every backward
program sums ``dw`` in FP32: each chunk's
partial row, then the fixed-order fold of the rows, one FP32 word per sum, except the
chunk sums of the windows the package README marks compensated, which add each block
of rows into an FP32 pair (a sum and its compensation word); there the FP32-pair
program is the faster one.  The backward accuracy guard has no edge: ``ACCURACY_EDGES``
is empty and every device, whatever its SM count against ``ACCURACY_MIN_SMS``, selects
the windows by ``T``.
Measured on B200 (sm_100a), GB300 (sm_103a) and R200 (sm_107a).
"""

FORWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships several programs of
one kernel and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  Within one window a row's ``y``, ``rstd`` and ``dx`` do
not depend on ``T``: two calls whose token counts fall in the same forward window (and,
for the backward, the same backward window) return bitwise-identical results on their
common rows.  Across a forward edge between programs of different per-row thread groups
the sum of squares is reduced in another order.  On the rows both programs compute,
``rstd`` then moves by at most two FP32 ulps (three on SM107 at the residual route's
edge 1024) on 10.8-18.2 % of the rows (24.0-31.6 % on the residual route); ``y`` moves
by one BF16 ulp, only on rows whose ``rstd`` moved, on at most 2.34 % of the rows
(4.30 % on the residual route); ``dx`` follows on those rows and the residual output
``h_new`` does not change.  Across every other measured forward edge, every measured
edge of H = 256 and H = 5120 among them, the outputs are bitwise identical.  The
programs of the smallest token counts that form ``rstd`` with the approximate reciprocal
square root (``rsqrt.approx``) give other ``rstd`` and ``y`` bits than the programs
beside them, so the outputs can change bits at their edges; ``EDGES`` marks those edges
as not yet measured.  Measured over three random inputs per edge, with BF16 and FP32
weights: on B200 (sm_100a) at most two ulps on up to 18.0 % of the rows (27.3 % on the
residual route), ``y`` on up to 2.03 % (2.83 %); on GB300 (sm_103a) at most two ulps on
up to 17.3 % (27.8 %), ``y`` on up to 2.03 % (2.34 %); on R200 (sm_107a, 212- and 216-SM
boards) at most three ulps on up to 18.2 % (31.6 %), ``y`` on up to 2.34 % (4.30 %).
Across a measured backward-only edge ``dx`` is bitwise identical but at the sm_107a
H = 6144 edges at 1664 and 3072 tokens, where the programs on the two sides (the
column-task and the one-row-per-stage program at 1664, the one-row-per-stage and the
bulk-copy program at 3072) reduce a row in different orders: ``dx`` differs on 23 of
the 1664 and on 69 of the 3072 common rows of the input measured, by at most two and
six BF16 bit-pattern steps, while ``y`` and ``rstd`` are identical.  The H = 6144
route, which switched between the base backward kernel and the bulk-copy one with
``dx`` differing on about 1-2 % of the rows, now runs column-task programs up to 384
tokens on sm_100a, 256 on sm_103a and 1664 on sm_107a, one-row-per-stage programs over
(384, 4096] on sm_100a and (256, 4096] on sm_103a (BF16 weight; the FP32-weight route
runs the bulk-copy kernel there) and over (1664, 3072] on sm_107a, and the bulk-copy
kernel above.  The one-row programs at ``T`` = 1 (H = 512, 1024, 1280, 1536 and
2048) fold the row's dot product in their own fixed order, so ``dx`` may change bits at
their edge; ``EDGES`` marks the edges not yet measured, every edge of the H = 1280, 1024
and 128 routes among them. ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` it is bitwise reproducible on devices with the same SM
count.  The backward splits the rows over ``ctas_per_sm`` x SM count CTAs, so devices of
one architecture with different SM counts (B200 with 148 and GB200 with 152 SMs, R200
boards with 212 and 216 SMs) cut different chunk tables and can give different ``dw``
bits.  In the windows of its smallest token counts a backward route other than H = 256
(on sm_100a H = 256 too, to 87 rows; the H = 128 routes at every ``T``)
runs column-task programs instead: one CTA sums each column of ``dw`` over every row in
a fixed order in FP32, so there ``dw`` does not depend on the SM count.  Every backward
program sums ``dw`` in FP32: each chunk's
partial row, then the fixed-order fold of the rows, one FP32 word per sum, except the
chunk sums of the windows the package README marks compensated, which add each block
of rows into an FP32 pair (a sum and its compensation word); there the FP32-pair
program is the faster one.  The backward accuracy guard has no edge: ``ACCURACY_EDGES``
is empty and every device, whatever its SM count against ``ACCURACY_MIN_SMS``, selects
the windows by ``T``.
Measured on B200 (sm_100a), GB300 (sm_103a) and R200 (sm_107a).

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge: ``=`` bitwise-identical outputs on the common rows across it (the
package tests check them), ``~`` the forward's measured differences (``y`` by at most
one BF16 ulp, ``rstd`` by at most two FP32 ulps, three on R200 (216 SMs) at the
residual route's edge 1024), ``dx`` the backward kernel switch, ``?`` not yet measured.

The FP32-weight routes (``fwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows, with the same outcome at every edge, and are not
listed; a route re-cut for the FP32 weight (another twin in a window) is listed with
its own windows.

* ``sm_100a`` ``fwd_h1024``: (0, 2] 6e5ca1f4 ? (2, 4] 9e8a7007 ? (4, 512] 6e5ca1f4 ?
  (512, 1024] 9e8a7007 ? (1024, 2048] 6e5ca1f4 ? (2048, 4096] 9e8a7007 ?
  (4096, 32768] 6e5ca1f4 ? (32768, inf) 9e8a7007
* ``sm_100a`` ``fwd_w32_h1024``: (0, 2] 364ed18c ? (2, 4] d99009bf ? (4, 32] 364ed18c ?
  (32, 64] d99009bf ? (64, 512] 364ed18c ? (512, 1024] d99009bf ? (1024, 32768] 364ed18c
  ? (32768, inf) d99009bf
* ``sm_100a`` ``fwd_h128``: (0, 2] 3808022d ? (2, 4] 628267fc ? (4, 16384] 3808022d ?
  (16384, inf) 628267fc
* ``sm_100a`` ``fwd_w32_h128``: (0, 2] 513f4ffb ? (2, 4] 1ecf9c26 ? (4, 64] 513f4ffb ?
  (64, 128] 1ecf9c26 ? (128, 16384] 513f4ffb ? (16384, inf) 1ecf9c26
* ``sm_100a`` ``fwd_h1280``: (0, 16] 4d9988d4 ? (16, 32] 03a74dfe ? (32, 256] 4d9988d4 ?
  (256, 512] 03a74dfe ? (512, 8192] 4d9988d4 ? (8192, 24576] 39b9b41b ?
  (24576, 32768] 007e82b2 ? (32768, inf) 03b55160
* ``sm_100a`` ``fwd_w32_h1280``: (0, 1] 6258b6da ? (1, 4] c43709ca ? (4, 8] 6258b6da ?
  (8, 16] c43709ca ? (16, 256] 6258b6da ? (256, 1024] c43709ca ? (1024, 2048] 6258b6da ?
  (2048, 32768] 58c2fce1 ? (32768, inf) 99b209a3
* ``sm_100a`` ``fwd_h1536``: (0, 1536] 1ea88523 ~ (1536, 3584] 2fdbab2a =
  (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_100a`` ``fwd_h2048``: (0, 1] b44107e0 ? (1, 1024] a5f323db ?
  (1024, 2048] 780b9833 ~ (2048, 4096] b44107e0 ~ (4096, inf) 780b9833
* ``sm_100a`` ``fwd_h256``: (0, 1024] 76b9c06c ? (1024, 8192] 0489c784 =
  (8192, 40960] 1b284bae = (40960, 81920] 86d51eff = (81920, 114688] 1388ab77 =
  (114688, 196608] 466331ff = (196608, 327680] ed9483d8 = (327680, inf) b8f32dd4
* ``sm_100a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_100a`` ``fwd_h512``: (0, 1024] 8757f6da ~ (1024, 4096] 7673978d =
  (4096, 24576] 3d20ab4a = (24576, inf) 3430e5d9
* ``sm_100a`` ``fwd_h5120``: (0, 64] 6e12e073 ? (64, 640] 5eef28cd =
  (640, 6144] b5e173f5 = (6144, inf) 5eef28cd
* ``sm_100a`` ``fwd_w32_h5120``: (0, 64] c1819da8 ? (64, 640] ba28d2c0 =
  (640, 6144] 9ecb62c2 = (6144, 16384] b49de9b4 = (16384, inf) 9ecb62c2
* ``sm_100a`` ``fwd_h6144``: (0, 768] bad4db91 = (768, 1024] bad4db91 =
  (1024, inf) 288d7b1d
* ``sm_100a`` ``fwd_w32_h6144``: (0, 768] 2763cd5e = (768, 1024] 2763cd5e =
  (1024, 6144] 3d619cd1 = (6144, inf) 5f89e5fd
* ``sm_100a`` ``fwd_residual_h6144``: (0, 256] 8b473033 ? (256, 1024] 03bdf566 ~
  (1024, 4096] e646bbdb ? (4096, inf) 41805c58
* ``sm_103a`` ``fwd_h1024``: (0, 2048] 6e5ca1f4 ? (2048, 4096] 9e8a7007 ?
  (4096, 32768] 6e5ca1f4 ? (32768, inf) 9e8a7007
* ``sm_103a`` ``fwd_w32_h1024``: (0, 2] 364ed18c ? (2, 4] d99009bf ? (4, 32] 364ed18c ?
  (32, 64] d99009bf ? (64, 512] 364ed18c ? (512, 1024] d99009bf ? (1024, 32768] 364ed18c
  ? (32768, inf) d99009bf
* ``sm_103a`` ``fwd_h128``: (0, 16384] 3808022d ? (16384, inf) 628267fc
* ``sm_103a`` ``fwd_h1280``: (0, 256] 4d9988d4 ? (256, 1024] 03a74dfe ?
  (1024, 8192] 4d9988d4 ? (8192, 24576] 39b9b41b ? (24576, inf) 007e82b2
* ``sm_103a`` ``fwd_w32_h1280``: (0, 512] 6258b6da ? (512, 1024] c43709ca ?
  (1024, 2048] 6258b6da ? (2048, 8192] 99b209a3 ? (8192, 24576] 62e8e275 ?
  (24576, 32768] 58c2fce1 ? (32768, inf) 6258b6da
* ``sm_103a`` ``fwd_h1536``: (0, 1] 1ea88523 ? (1, 256] d5238547 ? (256, 1536] 1ea88523
  ~ (1536, 3584] 2fdbab2a = (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_103a`` ``fwd_h2048``: (0, 1024] c5a8099e = (1024, 4096] b44107e0 ~
  (4096, inf) 780b9833
* ``sm_103a`` ``fwd_h256``: (0, 24] 76b9c06c ? (24, 8192] 0489c784 =
  (8192, 20480] bbb05314 = (20480, 40960] 1b284bae = (40960, 81920] 86d51eff =
  (81920, 114688] 466331ff = (114688, 196608] 86d51eff = (196608, 327680] ed9483d8 =
  (327680, 368640] 37f66318 = (368640, inf) b8f32dd4
* ``sm_103a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 3072] d447b7f6 = (3072, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_103a`` ``fwd_h512``: (0, 4] 8757f6da ? (4, 1025] def53244 ? (1025, 4096] 7673978d
  = (4096, 16384] 3d20ab4a = (16384, inf) 3430e5d9
* ``sm_103a`` ``fwd_h5120``: (0, 64] 6e12e073 ? (64, 640] 5eef28cd =
  (640, 6144] b5e173f5 = (6144, inf) 5eef28cd
* ``sm_103a`` ``fwd_w32_h5120``: (0, 64] c1819da8 ? (64, 640] 1defffdf =
  (640, 6144] 9ecb62c2 = (6144, inf) 9ecb62c2
* ``sm_103a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, inf) 288d7b1d
* ``sm_103a`` ``fwd_w32_h6144``: (0, 1024] 2763cd5e = (1024, 6144] 2763cd5e =
  (6144, inf) 5f89e5fd
* ``sm_103a`` ``fwd_residual_h6144``: (0, 4096] 8b473033 ? (4096, inf) e646bbdb
* ``sm_107a`` ``fwd_h1024``: (0, 8] 6e5ca1f4 ? (8, 16] 9e8a7007 ? (16, 256] 6e5ca1f4 ?
  (256, 1024] 9e8a7007 ? (1024, 4096] 6e5ca1f4 ? (4096, 8192] 9e8a7007 ?
  (8192, 8447] cffbc5c8 ? (8447, 13312] 547372db ? (13312, 32768] 9e8a7007 ?
  (32768, inf) cffbc5c8
* ``sm_107a`` ``fwd_w32_h1024``: (0, 8] 364ed18c ? (8, 16] d99009bf ? (16, 512] 364ed18c
  ? (512, 1536] d99009bf ? (1536, 4096] 364ed18c ? (4096, 8192] d99009bf ?
  (8192, 12288] f5110367 ? (12288, 16384] 364ed18c ? (16384, 16895] 8bd2d836 ?
  (16895, 24576] d99009bf ? (24576, 32768] d99009bf ? (32768, inf) 008fb179
* ``sm_107a`` ``fwd_h128``: (0, 32] 3808022d ? (32, 64] 628267fc ? (64, 2048] 3808022d ?
  (2048, 4096] 7221de56 ? (4096, 24576] 3808022d ? (24576, 32768] 7221de56 ?
  (32768, inf) a05a2262
* ``sm_107a`` ``fwd_w32_h128``: (0, 32] 513f4ffb ? (32, 64] 1ecf9c26 ?
  (64, 2048] 513f4ffb ? (2048, 4096] 04ea6484 ? (4096, 16384] 513f4ffb ?
  (16384, 32768] 1ecf9c26 ? (32768, inf) 67e0dfc1
* ``sm_107a`` ``fwd_h1280``: (0, 1] 03a74dfe ? (1, 4] 4d9988d4 ? (4, 8] 03a74dfe ?
  (8, 16] 007e82b2 ? (16, 128] 4d9988d4 ? (128, 512] 03a74dfe ? (512, 1024] 007e82b2 ?
  (1024, 1536] 03b55160 ? (1536, 2048] 4d9988d4 ? (2048, 4096] 03a74dfe ?
  (4096, 12288] 007e82b2 ? (12288, 16384] 03b55160 ? (16384, inf) 007e82b2
* ``sm_107a`` ``fwd_w32_h1280``: (0, 4] 6258b6da ? (4, 8] c43709ca ? (8, 128] 6258b6da ?
  (128, 512] c43709ca ? (512, 1024] 58c2fce1 ? (1024, 1536] 62e8e275 ?
  (1536, 2048] 6258b6da ? (2048, 8192] 62e8e275 ? (8192, 8703] 99b209a3 ?
  (8703, 9984] 51ca3c87 ? (9984, 16384] 99b209a3 ? (16384, inf) 58c2fce1
* ``sm_107a`` ``fwd_h1536``: (0, 2048] 1ea88523 ~ (2048, 12288] 4cdc4c49 =
  (12288, inf) 682675bd
* ``sm_107a`` ``fwd_h2048``: (0, 1024] b44107e0 ~ (1024, 8192] a4da8513 =
  (8192, 12288] 0ad4d53c = (12288, 16384] 0ad4d53c = (16384, inf) bde56bd5
* ``sm_107a`` ``fwd_w32_h2048``: (0, 1024] 0da8e18d ~ (1024, 8192] a72c6ece =
  (8192, 12288] 595f5060 = (12288, 16384] 40210950 = (16384, inf) 40210950
* ``sm_107a`` ``fwd_h256``: (0, 6144] c5ca409f = (6144, 32768] bbb05314 =
  (32768, 98304] 466331ff = (98304, inf) 37f66318
* ``sm_107a`` ``fwd_h4096``: (0, 512] 8392d1a0 ~ (512, 1024] fd913ecd =
  (1024, 4096] d447b7f6 = (4096, 20480] bf933e16 = (20480, 49152] bf933e16 =
  (49152, inf) fd913ecd
* ``sm_107a`` ``fwd_w32_h4096``: (0, 512] 4371270e ~ (512, 1024] f0db3068 =
  (1024, 4096] f9f3423b = (4096, 20480] f9f3423b = (20480, 49152] f9f3423b =
  (49152, inf) f9f3423b
* ``sm_107a`` ``fwd_h512``: (0, 256] 7673978d = (256, 6144] 2a42e5c0 =
  (6144, inf) 60f47c5d
* ``sm_107a`` ``fwd_h5120``: (0, 512] fd1097c4 = (512, 1152] 5eef28cd =
  (1152, 6144] b5e173f5 = (6144, 20480] 044d4a2a = (20480, 49152] 044d4a2a =
  (49152, inf) 5eef28cd
* ``sm_107a`` ``fwd_w32_h5120``: (0, 512] ba28d2c0 = (512, 1152] 1defffdf =
  (1152, 6144] 9ecb62c2 = (6144, 20480] b49de9b4 = (20480, 49152] b49de9b4 =
  (49152, inf) b49de9b4
* ``sm_107a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, 4096] cf1d0554 =
  (4096, inf) e7f4221a
* ``sm_107a`` ``fwd_residual_h6144``: (0, 256] e646bbdb ~ (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
"""

BACKWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships several programs of
one kernel and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  Within one window a row's ``y``, ``rstd`` and ``dx`` do
not depend on ``T``: two calls whose token counts fall in the same forward window (and,
for the backward, the same backward window) return bitwise-identical results on their
common rows.  Across a forward edge between programs of different per-row thread groups
the sum of squares is reduced in another order.  On the rows both programs compute,
``rstd`` then moves by at most two FP32 ulps (three on SM107 at the residual route's
edge 1024) on 10.8-18.2 % of the rows (24.0-31.6 % on the residual route); ``y`` moves
by one BF16 ulp, only on rows whose ``rstd`` moved, on at most 2.34 % of the rows
(4.30 % on the residual route); ``dx`` follows on those rows and the residual output
``h_new`` does not change.  Across every other measured forward edge, every measured
edge of H = 256 and H = 5120 among them, the outputs are bitwise identical.  The
programs of the smallest token counts that form ``rstd`` with the approximate reciprocal
square root (``rsqrt.approx``) give other ``rstd`` and ``y`` bits than the programs
beside them, so the outputs can change bits at their edges; ``EDGES`` marks those edges
as not yet measured.  Measured over three random inputs per edge, with BF16 and FP32
weights: on B200 (sm_100a) at most two ulps on up to 18.0 % of the rows (27.3 % on the
residual route), ``y`` on up to 2.03 % (2.83 %); on GB300 (sm_103a) at most two ulps on
up to 17.3 % (27.8 %), ``y`` on up to 2.03 % (2.34 %); on R200 (sm_107a, 212- and 216-SM
boards) at most three ulps on up to 18.2 % (31.6 %), ``y`` on up to 2.34 % (4.30 %).
Across a measured backward-only edge ``dx`` is bitwise identical but at the sm_107a
H = 6144 edges at 1664 and 3072 tokens, where the programs on the two sides (the
column-task and the one-row-per-stage program at 1664, the one-row-per-stage and the
bulk-copy program at 3072) reduce a row in different orders: ``dx`` differs on 23 of
the 1664 and on 69 of the 3072 common rows of the input measured, by at most two and
six BF16 bit-pattern steps, while ``y`` and ``rstd`` are identical.  The H = 6144
route, which switched between the base backward kernel and the bulk-copy one with
``dx`` differing on about 1-2 % of the rows, now runs column-task programs up to 384
tokens on sm_100a, 256 on sm_103a and 1664 on sm_107a, one-row-per-stage programs over
(384, 4096] on sm_100a and (256, 4096] on sm_103a (BF16 weight; the FP32-weight route
runs the bulk-copy kernel there) and over (1664, 3072] on sm_107a, and the bulk-copy
kernel above.  The one-row programs at ``T`` = 1 (H = 512, 1024, 1280, 1536 and
2048) fold the row's dot product in their own fixed order, so ``dx`` may change bits at
their edge; ``EDGES`` marks the edges not yet measured, every edge of the H = 1280, 1024
and 128 routes among them. ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` it is bitwise reproducible on devices with the same SM
count.  The backward splits the rows over ``ctas_per_sm`` x SM count CTAs, so devices of
one architecture with different SM counts (B200 with 148 and GB200 with 152 SMs, R200
boards with 212 and 216 SMs) cut different chunk tables and can give different ``dw``
bits.  In the windows of its smallest token counts a backward route other than H = 256
(on sm_100a H = 256 too, to 87 rows; the H = 128 routes at every ``T``)
runs column-task programs instead: one CTA sums each column of ``dw`` over every row in
a fixed order in FP32, so there ``dw`` does not depend on the SM count.  Every backward
program sums ``dw`` in FP32: each chunk's
partial row, then the fixed-order fold of the rows, one FP32 word per sum, except the
chunk sums of the windows the package README marks compensated, which add each block
of rows into an FP32 pair (a sum and its compensation word); there the FP32-pair
program is the faster one.  The backward accuracy guard has no edge: ``ACCURACY_EDGES``
is empty and every device, whatever its SM count against ``ACCURACY_MIN_SMS``, selects
the windows by ``T``.
Measured on B200 (sm_100a), GB300 (sm_103a) and R200 (sm_107a).

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge: ``=`` bitwise-identical outputs on the common rows across it (the
package tests check them), ``~`` the forward's measured differences (``y`` by at most
one BF16 ulp, ``rstd`` by at most two FP32 ulps, three on R200 (216 SMs) at the
residual route's edge 1024), ``dx`` the backward kernel switch, ``?`` not yet measured.

The FP32-weight routes (``bwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows, with the same outcome at every edge, and are not
listed; a route re-cut for the FP32 weight (another twin in a window) is listed with
its own windows.

* ``sm_100a`` ``bwd_h1024``: (0, 1] 2dc3fb26 ? (1, 64] 89fa3e96 ? (64, 128] 5ad5f096 ?
  (128, 1024] 89fa3e96 ? (1024, 8192] 0c581a13 ? (8192, inf) acd331db
* ``sm_100a`` ``bwd_w32_h1024``: (0, 1] 153f8119 ? (1, 1024] 744d7429 ?
  (1024, 8192] 561bea88 ? (8192, inf) b201ba67
* ``sm_100a`` ``bwd_h128``: (0, 1024] 93466008 ? (1024, 2048] 401a018c ?
  (2048, 11201] d197bcc3 ? (11201, inf) 401a018c
* ``sm_100a`` ``bwd_h1280``: (0, 1] 8895b022 ? (1, 128] a5f2942d ? (128, 512] 809235a3 ?
  (512, 4096] 12bbedff ? (4096, inf) 72a918b0
* ``sm_100a`` ``bwd_h1536``: (0, 1] f6233ff8 ? (1, 768] 4176a24f ? (768, 1024] 7dc29629
  = (1024, 1184] c1a9efdf ? (1184, 161792] 38e32d90 ? (161792, inf) b457c4d1
* ``sm_100a`` ``bwd_h2048``: (0, 1] b13b0c73 ? (1, 768] 923e668c ? (768, 1024] cc26e316
  ? (1024, 2047] c73074cf ? (2047, 22528] cc26e316 ? (22528, 58880] ceeef043 ?
  (58880, inf) ceeef043
* ``sm_100a`` ``bwd_w32_h2048``: (0, 1] d22b3e8c ? (1, 768] 9030dbeb ?
  (768, 1024] 07f9f2fb ? (1024, 16384] b187e598 = (16384, 58880] 7e6a0b63 ?
  (58880, inf) 7e6a0b63
* ``sm_100a`` ``bwd_h256``: (0, 87] 6959ea32 ? (87, 2048] a385ae16 =
  (2048, 5120] af005462 = (5120, 12288] a385ae16 ? (12288, 262144] 6bdbfe61 ?
  (262144, 1572864] a385ae16 = (1572864, inf) a385ae16
* ``sm_100a`` ``bwd_h4096``: (0, 203] 4822793e ? (203, 256] 3e644ea1 =
  (256, 1024] 59270fb2 ? (1024, 22528] 5572b267 ? (22528, 32768] 5abcdc29 ?
  (32768, 44032] 35a0b0ca = (44032, inf) 35a0b0ca
* ``sm_100a`` ``bwd_h512``: (0, 1] f283bf47 ? (1, 227] 931757c8 ? (227, 1024] 09601d42 ~
  (1024, 4096] 123db0b2 = (4096, 32768] ad1a5a3c = (32768, inf) 8877322a
* ``sm_100a`` ``bwd_h5120``: (0, 1] 080e0a6c ? (1, 256] dd3a3573 ? (256, 36864] bfde9b88
  ? (36864, 50688] b482c84d = (50688, inf) b482c84d
* ``sm_100a`` ``bwd_w32_h5120``: (0, 1] 747d86f3 ? (1, 256] 747d86f3 ?
  (256, 36864] f9b2d42c ? (36864, 50688] 9479b129 ? (50688, inf) 9479b129
* ``sm_100a`` ``bwd_h6144``: (0, 384] f2271417 ? (384, 4096] 61421bc7 ?
  (4096, 42496] b93bf060 = (42496, inf) b93bf060
* ``sm_100a`` ``bwd_w32_h6144``: (0, 384] 8a280a7e ? (384, 4096] 41b80b6a =
  (4096, 42496] 62beb52f = (42496, inf) 62beb52f
* ``sm_100a`` ``bwd_residual_h6144``: (0, 1152] 864e84f3 ? (1152, 42496] c1a3be94 ?
  (42496, inf) 84385468
* ``sm_100a`` ``bwd_w32_residual_h6144``: (0, 1152] 71d36617 ? (1152, 42496] ab0e20fe ?
  (42496, inf) ab0e20fe
* ``sm_103a`` ``bwd_h1024``: (0, 1] 2dc3fb26 ? (1, 64] 89fa3e96 ? (64, 128] 5ad5f096 ?
  (128, 1024] 89fa3e96 ? (1024, 1536] 5e32dd88 ? (1536, 2048] 5e32dd88 ?
  (2048, 4096] 5e32dd88 ? (4096, 8192] 5e32dd88 ? (8192, 12288] 5e32dd88 ?
  (12288, inf) acd331db
* ``sm_103a`` ``bwd_w32_h1024``: (0, 1] 153f8119 ? (1, 1024] 744d7429 ?
  (1024, 1536] ba8feeec ? (1536, 2048] ba8feeec ? (2048, 12288] ba8feeec ?
  (12288, inf) b201ba67
* ``sm_103a`` ``bwd_h128``: (0, 1024] 93466008 ? (1024, 2048] 401a018c ?
  (2048, 12288] d197bcc3 ? (12288, inf) 401a018c
* ``sm_103a`` ``bwd_h1280``: (0, 1] 8895b022 ? (1, 128] a5f2942d ? (128, 512] 809235a3 ?
  (512, 2048] 12bbedff ? (2048, inf) c646e097
* ``sm_103a`` ``bwd_w32_h1280``: (0, 1] 103805d9 ? (1, 128] 8d8ef930 ?
  (128, 512] e476d0b2 ? (512, 4096] 80956d25 ? (4096, inf) c66b8cbd
* ``sm_103a`` ``bwd_h1536``: (0, 1] e254714a ? (1, 31] 4aef8d1f ? (31, 214] 4aef8d1f ?
  (214, 1024] 7dc29629 = (1024, 1216] c1a9efdf ? (1216, 214016] 38e32d90 ?
  (214016, inf) 9cd3c766
* ``sm_103a`` ``bwd_h2048``: (0, 1] a1c71cc3 ? (1, 2] ed36a8d9 ? (2, 171] ed36a8d9 ?
  (171, 16384] cc26e316 ? (16384, 32768] caf6978e = (32768, 55296] 302f64db =
  (55296, inf) 302f64db
* ``sm_103a`` ``bwd_h256``: (0, 1680] a385ae16 ? (1680, 2048] a385ae16 ?
  (2048, 262144] 6bdbfe61 ? (262144, 720896] a385ae16 = (720896, 1572864] a385ae16 =
  (1572864, inf) a385ae16
* ``sm_103a`` ``bwd_h4096``: (0, 512] 5efbf78a ? (512, inf) 5572b267
* ``sm_103a`` ``bwd_w32_h4096``: (0, 1] 27615d0e ? (1, 512] ff7de8d3 ?
  (512, inf) 47a01388
* ``sm_103a`` ``bwd_h512``: (0, 1] f283bf47 ? (1, 4] bdfd7789 ? (4, 227] bdfd7789 ?
  (227, 1024] 09601d42 ? (1024, 4096] 8e7f4d81 = (4096, 32768] 274b0055 =
  (32768, inf) cfaae3ce
* ``sm_103a`` ``bwd_h5120``: (0, 64] dd3a3573 ? (64, 36544] bfde9b88 ?
  (36544, inf) 2fc4932f
* ``sm_103a`` ``bwd_h6144``: (0, 256] f2271417 ? (256, 4096] 61421bc7 ?
  (4096, 48640] 322574af = (48640, inf) 322574af
* ``sm_103a`` ``bwd_w32_h6144``: (0, 1] b3e6531e ? (1, 256] 8a280a7e ?
  (256, 4096] 41b80b6a = (4096, 48640] 9a8c0396 = (48640, inf) 9a8c0396
* ``sm_103a`` ``bwd_residual_h6144``: (0, 1216] 864e84f3 ? (1216, 48640] 04ebec87 ?
  (48640, inf) 04ebec87
* ``sm_107a`` ``bwd_h1024``: (0, 1] 2dc3fb26 ? (1, 8] 5ad5f096 ? (8, 1536] 89fa3e96 ?
  (1536, 2048] 0c581a13 ? (2048, 4095] 5e32dd88 ? (4095, inf) 4572a519
* ``sm_107a`` ``bwd_h128``: (0, 1024] 93466008 ? (1024, 2047] 401a018c ?
  (2047, 65536] d197bcc3 ? (65536, inf) 401a018c
* ``sm_107a`` ``bwd_h1280``: (0, 1] 8895b022 ? (1, 128] a5f2942d ? (128, 512] 809235a3 ?
  (512, 6143] 12bbedff ? (6143, 32767] 3f26352e ? (32767, inf) 7ac3bb64
* ``sm_107a`` ``bwd_w32_h1280``: (0, 1] 103805d9 ? (1, 128] 8d8ef930 ?
  (128, 512] e476d0b2 ? (512, 4096] 80956d25 ? (4096, 6143] c66b8cbd ?
  (6143, 32767] b1a35315 ? (32767, inf) 7a4606b0
* ``sm_107a`` ``bwd_h1536``: (0, 1] c3b1b46a ? (1, 576] 1207b056 ? (576, 1024] 7dc29629
  = (1024, 12288] 88510c60 = (12288, 16384] 6a428da2 = (16384, 45056] 56d26996 =
  (45056, inf) 411e270c
* ``sm_107a`` ``bwd_h2048``: (0, 1] 92def3f9 ? (1, 256] 6720b4a1 ? (256, 1024] 845cdcd8
  ~ (1024, 16384] cc26e316 = (16384, 50688] 845cdcd8 = (50688, inf) 21ee6fc9
* ``sm_107a`` ``bwd_h256``: (0, 1572864] a385ae16 = (1572864, inf) a385ae16
* ``sm_107a`` ``bwd_h4096``: (0, 256] c1de7e41 ? (256, 1024] a647fccc =
  (1024, 4096] d41a514f = (4096, 24064] ee4a9053 ? (24064, 50176] a647fccc ?
  (50176, inf) e9e21bbe
* ``sm_107a`` ``bwd_w32_h4096``: (0, 256] 30ec8bb6 ? (256, 1024] dc439f83 =
  (1024, 4096] b0863320 = (4096, 24064] 0f1859b9 = (24064, 50176] 0ff440da =
  (50176, inf) 0ff440da
* ``sm_107a`` ``bwd_h512``: (0, 1] f283bf47 ? (1, 768] b143f616 ? (768, 1024] 3feed502 =
  (1024, 4096] 3feed502 = (4096, 16384] dfaf378a ? (16384, inf) eb141cc9
* ``sm_107a`` ``bwd_h5120``: (0, 1] 080e0a6c ? (1, 256] dd3a3573 ? (256, 1280] 080e0a6c
  ? (1280, 4096] 838c077d ? (4096, 19968] 6816fb80 ? (19968, inf) 47da668e
* ``sm_107a`` ``bwd_h6144``: (0, 1] 0dafdf09 ? (1, 256] dffe53d5 ? (256, 1664] 0dafdf09
  dx (1664, 3072] af220b26 dx (3072, 19456] 5d292172 = (19456, inf) 9cc51adf
* ``sm_107a`` ``bwd_residual_h6144``: (0, 368] ee1be07a ? (368, 19456] 36cad444 ?
  (19456, inf) 2f149999
"""

SECTIONS: dict[str, str] = {"fwd": FORWARD_SECTION, "bwd": BACKWARD_SECTION}

__all__ = [
    "ACCURACY_EDGES",
    "ACCURACY_MIN_SMS",
    "BACKWARD_SECTION",
    "EDGES",
    "EDGE_CLASSES",
    "FORWARD_SECTION",
    "NOTES",
    "PROGRAM_WINDOWS",
    "SECTIONS",
]
