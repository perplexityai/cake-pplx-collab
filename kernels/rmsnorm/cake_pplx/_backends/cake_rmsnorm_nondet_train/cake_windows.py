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

Program selection by token count of the Cake BF16 RMSNorm training kernels with the
non-deterministic weight gradient (SM100 / SM103 / SM107), rendered from the registry
of this package (:mod:`.cake_jit`) by its exporter; do not edit.  ``PROGRAM_WINDOWS``
is the table ``cake_jit.program_windows`` derives from the registry's ``MODULES`` and
``ROUTES`` at call time; ``EDGES`` the measured outcome at every window edge; ``NOTES``
the contract of the outputs across the windows; ``SECTIONS`` the text the host module
appends to the forward and backward docstrings.
"""

from __future__ import annotations

# Per architecture and route (``cake_jit.route_key``): the windows ``above < T <= at_most``
# (``at_most`` None: unbounded) and the program each launches, from the smallest rows up.
PROGRAM_WINDOWS: dict[str, dict[str, tuple[tuple[int, int | None, str], ...]]] = {
    "sm_100a": {
        "fwd_h1024": (
            (0, 2, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (2, 4, "cake_rmsnorm_train_9e8a70075f28be558bba"),
            (4, 512, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (512, 1024, "cake_rmsnorm_train_9e8a70075f28be558bba"),
            (1024, 2048, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (2048, 4096, "cake_rmsnorm_train_9e8a70075f28be558bba"),
            (4096, 32768, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (32768, None, "cake_rmsnorm_train_9e8a70075f28be558bba"),
        ),
        "fwd_w32_h1024": (
            (0, 2, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (2, 4, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (4, 32, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (32, 64, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (64, 512, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (512, 1024, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (1024, 32768, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (32768, None, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
        ),
        "fwd_h128": (
            (0, 2, "cake_rmsnorm_train_3808022dd402387eed16"),
            (2, 4, "cake_rmsnorm_train_628267fc724c2c8f0e07"),
            (4, 16384, "cake_rmsnorm_train_3808022dd402387eed16"),
            (16384, None, "cake_rmsnorm_train_628267fc724c2c8f0e07"),
        ),
        "fwd_w32_h128": (
            (0, 2, "cake_rmsnorm_train_513f4ffbfefd9d019863"),
            (2, 4, "cake_rmsnorm_train_1ecf9c260e418bc70046"),
            (4, 64, "cake_rmsnorm_train_513f4ffbfefd9d019863"),
            (64, 128, "cake_rmsnorm_train_1ecf9c260e418bc70046"),
            (128, 16384, "cake_rmsnorm_train_513f4ffbfefd9d019863"),
            (16384, None, "cake_rmsnorm_train_1ecf9c260e418bc70046"),
        ),
        "fwd_h1280": (
            (0, 16, "cake_rmsnorm_train_4d9988d4680c7c3c02a5"),
            (16, 32, "cake_rmsnorm_train_03a74dfe62df95c2a752"),
            (32, 256, "cake_rmsnorm_train_4d9988d4680c7c3c02a5"),
            (256, 512, "cake_rmsnorm_train_03a74dfe62df95c2a752"),
            (512, 8192, "cake_rmsnorm_train_4d9988d4680c7c3c02a5"),
            (8192, 24576, "cake_rmsnorm_train_39b9b41b16769252153d"),
            (24576, 32768, "cake_rmsnorm_train_007e82b24390c88156ac"),
            (32768, None, "cake_rmsnorm_train_03b5516039c533b453b9"),
        ),
        "fwd_w32_h1280": (
            (0, 1, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (1, 4, "cake_rmsnorm_train_c43709cadf31ed04033c"),
            (4, 8, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (8, 16, "cake_rmsnorm_train_c43709cadf31ed04033c"),
            (16, 256, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (256, 1024, "cake_rmsnorm_train_c43709cadf31ed04033c"),
            (1024, 2048, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (2048, 32768, "cake_rmsnorm_train_58c2fce1ea0b40758e91"),
            (32768, None, "cake_rmsnorm_train_99b209a37f0203af57ce"),
        ),
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_b44107e05796c2922187"),
            (1024, 2048, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
            (2048, 4096, "cake_rmsnorm_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (1024, 2048, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
            (2048, 4096, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 8192, "cake_rmsnorm_train_0489c784c13bb25499f6"),
            (8192, 40960, "cake_rmsnorm_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_train_1388ab779a35c4e41efb"),
            (114688, 196608, "cake_rmsnorm_train_466331ff3cd2d6c62738"),
            (196608, 327680, "cake_rmsnorm_train_ed9483d8c356affee55d"),
            (327680, None, "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 8192, "cake_rmsnorm_train_f2e721db4c30900f12fe"),
            (8192, 40960, "cake_rmsnorm_train_77456435c9fbd1d18fcf"),
            (40960, 81920, "cake_rmsnorm_train_42d82d26e97c36651dae"),
            (81920, 114688, "cake_rmsnorm_train_b0299f6e0881fb35f27b"),
            (114688, 196608, "cake_rmsnorm_train_32bebe5a7782311d60a0"),
            (196608, 327680, "cake_rmsnorm_train_d0a656f053396a58c49a"),
            (327680, None, "cake_rmsnorm_train_670bfb3eed0ddf6ba461"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 6144, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 6144, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_train_7673978dc698790054d2"),
            (4096, 24576, "cake_rmsnorm_train_3d20ab4a432bddf7349a"),
            (24576, None, "cake_rmsnorm_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (4096, 24576, "cake_rmsnorm_train_7da36ff557bb2bac8d89"),
            (24576, None, "cake_rmsnorm_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 640, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 640, "cake_rmsnorm_train_ba28d2c0c2c81a2c714e"),
            (640, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, 16384, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
            (16384, None, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
        ),
        "fwd_h6144": (
            (0, 768, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (768, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 768, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (768, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_train_3d619cd1e3ae638cce3d"),
            (6144, None, "cake_rmsnorm_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_308da18230017dadc02d"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_8e3407e059cfc005f421"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h128": (
            (0, 1024, "cake_rmsnorm_train_9346600882e53a589509"),
            (1024, 2048, "cake_rmsnorm_train_401a018c68bc082078b2"),
            (2048, 8192, "cake_rmsnorm_train_d197bcc34403e94ee2bd"),
            (8192, 12288, "cake_rmsnorm_train_8b0b86c827709e73ee26"),
            (12288, 16384, "cake_rmsnorm_train_c50d3bbd794af91a2538"),
            (16384, None, "cake_rmsnorm_train_8b0b86c827709e73ee26"),
        ),
        "bwd_w32_h128": (
            (0, 1024, "cake_rmsnorm_train_ab8a320c36b909e7bf1e"),
            (1024, 1536, "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea"),
            (1536, 2048, "cake_rmsnorm_train_a45d09aebf1bdca80e95"),
            (2048, 8192, "cake_rmsnorm_train_e23083c7af388f9e2f50"),
            (8192, 12288, "cake_rmsnorm_train_3e751b6f0ebbd6be9481"),
            (12288, 16384, "cake_rmsnorm_train_d0fe504e1d8611c0b2dc"),
            (16384, None, "cake_rmsnorm_train_3e751b6f0ebbd6be9481"),
        ),
        "bwd_h256": (
            (0, 4, "cake_rmsnorm_train_780c64c6f2c4615bfbcf"),
            (4, 24, "cake_rmsnorm_train_f2ccf7197a641c39acbf"),
            (24, 64, "cake_rmsnorm_train_9fb6b410e91f9d92becf"),
            (64, 20480, "cake_rmsnorm_train_195c4317b51bb4ea1137"),
            (20480, 92160, "cake_rmsnorm_train_44ef314181f3f2e93293"),
            (92160, 524288, "cake_rmsnorm_train_a385ae1681a2c078344e"),
            (524288, None, "cake_rmsnorm_train_44ef314181f3f2e93293"),
        ),
        "bwd_w32_h256": (
            (0, 4, "cake_rmsnorm_train_67cb56412bd46304b711"),
            (4, 24, "cake_rmsnorm_train_aa745395d7542d8a8e3a"),
            (24, 64, "cake_rmsnorm_train_6a0114cb0b27a780a4c3"),
            (64, 20480, "cake_rmsnorm_train_8de0131445eed6eeabb7"),
            (20480, 92160, "cake_rmsnorm_train_b71bbe9852c16fe03114"),
            (92160, 864240, "cake_rmsnorm_train_420788e663a2c3a810f6"),
            (864240, None, "cake_rmsnorm_train_b71bbe9852c16fe03114"),
        ),
        "bwd_h512": (
            (0, 4, "cake_rmsnorm_train_64a299691c17079fe510"),
            (4, 16, "cake_rmsnorm_train_c167e0d26b363cf19e7b"),
            (16, 227, "cake_rmsnorm_train_931757c844a9d257c08b"),
            (227, 1024, "cake_rmsnorm_train_09601d42628de24f6f87"),
            (1024, 1280, "cake_rmsnorm_train_123db0b2f7fb38cfab03"),
            (1280, 4096, "cake_rmsnorm_train_d0bde38aa443d42a5f04"),
            (4096, 32768, "cake_rmsnorm_train_ad1a5a3c5f88e3d10036"),
            (32768, None, "cake_rmsnorm_train_8877322a3002e3a4fefd"),
        ),
        "bwd_w32_h512": (
            (0, 4, "cake_rmsnorm_train_8c261dfc6e34351d7a11"),
            (4, 16, "cake_rmsnorm_train_88e6252beaf7d056d555"),
            (16, 227, "cake_rmsnorm_train_404c8f54ab3129877350"),
            (227, 292, "cake_rmsnorm_train_1a322bb1c550a7d70638"),
            (292, 960, "cake_rmsnorm_train_9f07cea1d13b923ed91e"),
            (960, None, "cake_rmsnorm_train_fd08c6f436eb6071fa22"),
        ),
        "bwd_h1024": (
            (0, 4, "cake_rmsnorm_train_330930e9329d1e53aa5c"),
            (4, 8, "cake_rmsnorm_train_28647b7ceb414d4fd02c"),
            (8, 64, "cake_rmsnorm_train_89fa3e964f6ac26dc6a2"),
            (64, 128, "cake_rmsnorm_train_5ad5f0962d5c5e8f63ac"),
            (128, 1024, "cake_rmsnorm_train_89fa3e964f6ac26dc6a2"),
            (1024, 2048, "cake_rmsnorm_train_0c581a133624ff396ab4"),
            (2048, 32768, "cake_rmsnorm_train_af798d1034849cc692e9"),
            (32768, None, "cake_rmsnorm_train_acd331db1201bf2fb44e"),
        ),
        "bwd_w32_h1024": (
            (0, 4, "cake_rmsnorm_train_81e742d5d2cf1e0f790a"),
            (4, 8, "cake_rmsnorm_train_8501e95ea685c476cb50"),
            (8, 1024, "cake_rmsnorm_train_744d74299c2d45867c94"),
            (1024, 2048, "cake_rmsnorm_train_561bea88a25c2f245c29"),
            (2048, 4096, "cake_rmsnorm_train_1a45ade5011422a0d019"),
            (4096, 12288, "cake_rmsnorm_train_1a45ade5011422a0d019"),
            (12288, 32768, "cake_rmsnorm_train_1a45ade5011422a0d019"),
            (32768, None, "cake_rmsnorm_train_b201ba67ac366efbd7be"),
        ),
        "bwd_h1280": (
            (0, 1, "cake_rmsnorm_train_8895b0224bb3d596a3a6"),
            (1, 128, "cake_rmsnorm_train_a5f2942d61e591f57a76"),
            (128, 512, "cake_rmsnorm_train_809235a3965d337a6877"),
            (512, 4096, "cake_rmsnorm_train_12bbedff93fcaafa475d"),
            (4096, 8192, "cake_rmsnorm_train_c646e0976e08c6fe8a91"),
            (8192, None, "cake_rmsnorm_train_b2cf87e64c9399e38407"),
        ),
        "bwd_w32_h1280": (
            (0, 1, "cake_rmsnorm_train_103805d93149c8668264"),
            (1, 128, "cake_rmsnorm_train_8d8ef930b7d650c5538a"),
            (128, 512, "cake_rmsnorm_train_e476d0b2c0bd4285f16d"),
            (512, 4096, "cake_rmsnorm_train_80956d258172dfa76c2a"),
            (4096, None, "cake_rmsnorm_train_f6956a41e69eb126197e"),
        ),
        "bwd_h1536": (
            (0, 2, "cake_rmsnorm_train_db218c92c4c86b9da85a"),
            (2, 64, "cake_rmsnorm_train_02e761bcf94053af6fc3"),
            (64, 384, "cake_rmsnorm_train_dc478ac1195882aedee6"),
            (384, 1024, "cake_rmsnorm_train_7dc29629ff21625607f8"),
            (1024, 1184, "cake_rmsnorm_train_c1a9efdf06c3087b4efe"),
            (1184, 77824, "cake_rmsnorm_train_38e32d90f62f7cfc7b51"),
            (77824, 78784, "cake_rmsnorm_train_b457c4d134e44476dc6f"),
            (78784, 131072, "cake_rmsnorm_train_0d5dc239536f113f4926"),
            (131072, 214016, "cake_rmsnorm_train_38e32d90f62f7cfc7b51"),
            (214016, None, "cake_rmsnorm_train_ae610afdf77ff6714a5b"),
        ),
        "bwd_w32_h1536": (
            (0, 2, "cake_rmsnorm_train_6d3986ee0dd962cafd21"),
            (2, 64, "cake_rmsnorm_train_166e288c9192be6b994c"),
            (64, 384, "cake_rmsnorm_train_e83ee45f66352eaef755"),
            (384, 1024, "cake_rmsnorm_train_baff9ecbafe7de1e6030"),
            (1024, 1184, "cake_rmsnorm_train_4b6981c10fd40a9d8aec"),
            (1184, 77824, "cake_rmsnorm_train_e30c1899e859da4f4c68"),
            (77824, 131072, "cake_rmsnorm_train_363eabb3661dab78f3d3"),
            (131072, 214016, "cake_rmsnorm_train_e30c1899e859da4f4c68"),
            (214016, None, "cake_rmsnorm_train_4b6981c10fd40a9d8aec"),
        ),
        "bwd_h2048": (
            (0, 4, "cake_rmsnorm_train_c5f62407b828a85f82ec"),
            (4, 12, "cake_rmsnorm_train_a951623c157d070e85ab"),
            (12, 64, "cake_rmsnorm_train_518cf16a7bf8e88042ee"),
            (64, 336, "cake_rmsnorm_train_64e6aa82bb2c6d4ea978"),
            (336, 3072, "cake_rmsnorm_train_caf6978ed0a17fbf65b9"),
            (3072, 4096, "cake_rmsnorm_train_9ac8d21e46a658e7913f"),
            (4096, 7168, "cake_rmsnorm_train_c73074cf44f07fcf3bb6"),
            (7168, 11264, "cake_rmsnorm_train_0fa4f255caf709d09f66"),
            (11264, 16384, "cake_rmsnorm_train_c73074cf44f07fcf3bb6"),
            (16384, 58880, "cake_rmsnorm_train_ceeef043ac64f37fc529"),
            (58880, 131072, "cake_rmsnorm_train_8404aa4380f6db7259cc"),
            (131072, None, "cake_rmsnorm_train_0fa4f255caf709d09f66"),
        ),
        "bwd_w32_h2048": (
            (0, 4, "cake_rmsnorm_train_0017ab41f68d8689ebc2"),
            (4, 14, "cake_rmsnorm_train_65e5f2024d26a36f2e18"),
            (14, 64, "cake_rmsnorm_train_c2f08c957d4b0f17dddd"),
            (64, 336, "cake_rmsnorm_train_9030dbeb948761ec2258"),
            (336, 1024, "cake_rmsnorm_train_17493054fed5409b6df9"),
            (1024, 4096, "cake_rmsnorm_train_ba7dbf6a07744268705e"),
            (4096, 15403, "cake_rmsnorm_train_b187e5989302def22b2d"),
            (15403, 45056, "cake_rmsnorm_train_ba7dbf6a07744268705e"),
            (45056, 58880, "cake_rmsnorm_train_ba7dbf6a07744268705e"),
            (58880, 65536, "cake_rmsnorm_train_7e6a0b631aea3edf51f9"),
            (65536, 131072, "cake_rmsnorm_train_54f6e4b8d937f7fb5619"),
            (131072, None, "cake_rmsnorm_train_ba7dbf6a07744268705e"),
        ),
        "bwd_h4096": (
            (0, 4, "cake_rmsnorm_train_945a0afdb82ab47eb643"),
            (4, 12, "cake_rmsnorm_train_41f44b4bc0b765350a50"),
            (12, 64, "cake_rmsnorm_train_82b84544dba8e2b70bc1"),
            (64, 1152, "cake_rmsnorm_train_4822793ea0c954c11af6"),
            (1152, 4096, "cake_rmsnorm_train_9a830956447d4e304914"),
            (4096, 4608, "cake_rmsnorm_train_35a0b0ca6b2060313255"),
            (4608, 16384, "cake_rmsnorm_train_cf4c213e50482afdacdb"),
            (16384, 36864, "cake_rmsnorm_train_681b33d9c9d0ac4d29a4"),
            (36864, 44032, "cake_rmsnorm_train_35a0b0ca6b2060313255"),
            (44032, 65536, "cake_rmsnorm_train_35a0b0ca6b2060313255"),
            (65536, 131072, "cake_rmsnorm_train_cda582bb216e6776b449"),
            (131072, None, "cake_rmsnorm_train_15e157d2ee9080d206a9"),
        ),
        "bwd_w32_h4096": (
            (0, 4, "cake_rmsnorm_train_f6bd1634a115bd281055"),
            (4, 14, "cake_rmsnorm_train_b378ba007691c582171e"),
            (14, 64, "cake_rmsnorm_train_cdef0b34f5080dbc61d1"),
            (64, 1792, "cake_rmsnorm_train_8616070567821eaa3e35"),
            (1792, 3584, "cake_rmsnorm_train_f7573bcb748c0784f3ec"),
            (3584, 36864, "cake_rmsnorm_train_f7573bcb748c0784f3ec"),
            (36864, 44032, "cake_rmsnorm_train_626aed743358c27513f1"),
            (44032, 65536, "cake_rmsnorm_train_626aed743358c27513f1"),
            (65536, 131072, "cake_rmsnorm_train_3a10d379e6a7ae778964"),
            (131072, None, "cake_rmsnorm_train_f7573bcb748c0784f3ec"),
        ),
        "bwd_h5120": (
            (0, 4, "cake_rmsnorm_train_dbc680f356be4cd643de"),
            (4, 64, "cake_rmsnorm_train_29bcbde1d271ea989fa8"),
            (64, 767, "cake_rmsnorm_train_080e0a6c3b18e9bad790"),
            (767, 36864, "cake_rmsnorm_train_bfde9b88e800fc34722c"),
            (36864, 56256, "cake_rmsnorm_train_b482c84dbffcd75a4a4c"),
            (56256, 131072, "cake_rmsnorm_train_670f4cafe8c34a429d30"),
            (131072, None, "cake_rmsnorm_train_bfde9b88e800fc34722c"),
        ),
        "bwd_w32_h5120": (
            (0, 4, "cake_rmsnorm_train_d47138228b123e55fe5d"),
            (4, 64, "cake_rmsnorm_train_cb6539573725ac861f94"),
            (64, 767, "cake_rmsnorm_train_139d162a8bb29d1b2d35"),
            (767, 36864, "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48"),
            (36864, 56256, "cake_rmsnorm_train_9479b1298a1b19d4d996"),
            (56256, 131072, "cake_rmsnorm_train_13bd0c5f56e37867eb26"),
            (131072, None, "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48"),
        ),
        "bwd_h6144": (
            (0, 4, "cake_rmsnorm_train_e37752b056dd4e4a0bee"),
            (4, 64, "cake_rmsnorm_train_e6475d8fa504ef4d7d2d"),
            (64, 1152, "cake_rmsnorm_train_44508df7dab2ab688f85"),
            (1152, 4096, "cake_rmsnorm_train_5c804aed7ad14b99e1d0"),
            (4096, 42496, "cake_rmsnorm_train_b93bf060d4d3defc8150"),
            (42496, 65536, "cake_rmsnorm_train_9433061da18b2db4299d"),
            (65536, 131072, "cake_rmsnorm_train_301133b646c4197fe462"),
            (131072, None, "cake_rmsnorm_train_9433061da18b2db4299d"),
        ),
        "bwd_residual_h6144": (
            (0, 2, "cake_rmsnorm_train_deeb187630473818923d"),
            (2, 64, "cake_rmsnorm_train_3a55646d574035bfa271"),
            (64, 1408, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (1408, 65536, "cake_rmsnorm_train_f6537c86c7ffae183a50"),
            (65536, 131072, "cake_rmsnorm_train_6feab384e4c85e893ed5"),
            (131072, None, "cake_rmsnorm_train_3fbfdd750cbeb47032b5"),
        ),
        "bwd_w32_h6144": (
            (0, 4, "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9"),
            (4, 64, "cake_rmsnorm_train_f4d41a0991647b82ac21"),
            (64, 1168, "cake_rmsnorm_train_b3e6531e46667901ab91"),
            (1168, 3584, "cake_rmsnorm_train_41b80b6af97974feb181"),
            (3584, 65536, "cake_rmsnorm_train_a562656cef5bf5dda160"),
            (65536, 131072, "cake_rmsnorm_train_c61519fe43c45818b9b4"),
            (131072, None, "cake_rmsnorm_train_a562656cef5bf5dda160"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 4, "cake_rmsnorm_train_a6b1a6e565dc98b49e1a"),
            (4, 64, "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c"),
            (64, 1408, "cake_rmsnorm_train_71d36617003ab7742a77"),
            (1408, 65536, "cake_rmsnorm_train_a640d0b799889f785ad5"),
            (65536, 131072, "cake_rmsnorm_train_f5453c09151f9d441744"),
            (131072, None, "cake_rmsnorm_train_a640d0b799889f785ad5"),
        ),
    },
    "sm_103a": {
        "fwd_h1024": (
            (0, 2048, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (2048, 4096, "cake_rmsnorm_train_9e8a70075f28be558bba"),
            (4096, 32768, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (32768, None, "cake_rmsnorm_train_9e8a70075f28be558bba"),
        ),
        "fwd_w32_h1024": (
            (0, 2, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (2, 4, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (4, 32, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (32, 64, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (64, 512, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (512, 1024, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (1024, 32768, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (32768, None, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
        ),
        "fwd_h128": (
            (0, 16384, "cake_rmsnorm_train_3808022dd402387eed16"),
            (16384, None, "cake_rmsnorm_train_628267fc724c2c8f0e07"),
        ),
        "fwd_w32_h128": (
            (0, 16384, "cake_rmsnorm_train_513f4ffbfefd9d019863"),
            (16384, None, "cake_rmsnorm_train_1ecf9c260e418bc70046"),
        ),
        "fwd_h1280": (
            (0, 256, "cake_rmsnorm_train_4d9988d4680c7c3c02a5"),
            (256, 1024, "cake_rmsnorm_train_03a74dfe62df95c2a752"),
            (1024, 8192, "cake_rmsnorm_train_4d9988d4680c7c3c02a5"),
            (8192, 24576, "cake_rmsnorm_train_39b9b41b16769252153d"),
            (24576, None, "cake_rmsnorm_train_007e82b24390c88156ac"),
        ),
        "fwd_w32_h1280": (
            (0, 512, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (512, 1024, "cake_rmsnorm_train_c43709cadf31ed04033c"),
            (1024, 2048, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (2048, 8192, "cake_rmsnorm_train_99b209a37f0203af57ce"),
            (8192, 24576, "cake_rmsnorm_train_62e8e2756e015c39bf5c"),
            (24576, 32768, "cake_rmsnorm_train_58c2fce1ea0b40758e91"),
            (32768, None, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
        ),
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_c5a8099e177e463daab9"),
            (1024, 4096, "cake_rmsnorm_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_01f0ebf3db9fa2c0884e"),
            (1024, 4096, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 8192, "cake_rmsnorm_train_0489c784c13bb25499f6"),
            (8192, 20480, "cake_rmsnorm_train_bbb05314930d192dfe7b"),
            (20480, 40960, "cake_rmsnorm_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_train_466331ff3cd2d6c62738"),
            (114688, 196608, "cake_rmsnorm_train_86d51eff64fc9fe57cf0"),
            (196608, 327680, "cake_rmsnorm_train_ed9483d8c356affee55d"),
            (327680, 368640, "cake_rmsnorm_train_37f663183cbff23c2893"),
            (368640, None, "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 8192, "cake_rmsnorm_train_f2e721db4c30900f12fe"),
            (8192, 20480, "cake_rmsnorm_train_a8fffd93682c5e382dc5"),
            (20480, 40960, "cake_rmsnorm_train_77456435c9fbd1d18fcf"),
            (40960, 81920, "cake_rmsnorm_train_42d82d26e97c36651dae"),
            (81920, 114688, "cake_rmsnorm_train_32bebe5a7782311d60a0"),
            (114688, 196608, "cake_rmsnorm_train_42d82d26e97c36651dae"),
            (196608, 327680, "cake_rmsnorm_train_d0a656f053396a58c49a"),
            (327680, 368640, "cake_rmsnorm_train_18e3c0a5204006b3a56f"),
            (368640, None, "cake_rmsnorm_train_670bfb3eed0ddf6ba461"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 3072, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (3072, 6144, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 3072, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (3072, 6144, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_train_7673978dc698790054d2"),
            (4096, 16384, "cake_rmsnorm_train_3d20ab4a432bddf7349a"),
            (16384, None, "cake_rmsnorm_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (4096, 16384, "cake_rmsnorm_train_7da36ff557bb2bac8d89"),
            (16384, None, "cake_rmsnorm_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 640, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 640, "cake_rmsnorm_train_1defffdf2712cae36784"),
            (640, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, None, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (6144, None, "cake_rmsnorm_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_308da18230017dadc02d"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_8e3407e059cfc005f421"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h128": (
            (0, 1024, "cake_rmsnorm_train_9346600882e53a589509"),
            (1024, 2048, "cake_rmsnorm_train_401a018c68bc082078b2"),
            (2048, 8192, "cake_rmsnorm_train_d197bcc34403e94ee2bd"),
            (8192, 12288, "cake_rmsnorm_train_8b0b86c827709e73ee26"),
            (12288, 16384, "cake_rmsnorm_train_c50d3bbd794af91a2538"),
            (16384, None, "cake_rmsnorm_train_8b0b86c827709e73ee26"),
        ),
        "bwd_w32_h128": (
            (0, 1024, "cake_rmsnorm_train_ab8a320c36b909e7bf1e"),
            (1024, 2048, "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea"),
            (2048, 8192, "cake_rmsnorm_train_e23083c7af388f9e2f50"),
            (8192, 12288, "cake_rmsnorm_train_3e751b6f0ebbd6be9481"),
            (12288, 16384, "cake_rmsnorm_train_d0fe504e1d8611c0b2dc"),
            (16384, None, "cake_rmsnorm_train_3e751b6f0ebbd6be9481"),
        ),
        "bwd_h256": (
            (0, 4, "cake_rmsnorm_train_780c64c6f2c4615bfbcf"),
            (4, 24, "cake_rmsnorm_train_f2ccf7197a641c39acbf"),
            (24, 64, "cake_rmsnorm_train_9fb6b410e91f9d92becf"),
            (64, 1080, "cake_rmsnorm_train_a385ae1681a2c078344e"),
            (1080, 2048, "cake_rmsnorm_train_af005462cd6a301f31c4"),
            (2048, 262144, "cake_rmsnorm_train_6bdbfe619b8270ff9a21"),
            (262144, 720896, "cake_rmsnorm_train_a385ae1681a2c078344e"),
            (720896, 1572864, "cake_rmsnorm_train_a385ae1681a2c078344e"),
            (1572864, None, "cake_rmsnorm_train_a385ae1681a2c078344e"),
        ),
        "bwd_w32_h256": (
            (0, 4, "cake_rmsnorm_train_67cb56412bd46304b711"),
            (4, 24, "cake_rmsnorm_train_aa745395d7542d8a8e3a"),
            (24, 64, "cake_rmsnorm_train_6a0114cb0b27a780a4c3"),
            (64, 1080, "cake_rmsnorm_train_420788e663a2c3a810f6"),
            (1080, 2048, "cake_rmsnorm_train_b7d9577ff884de98ea06"),
            (2048, 262144, "cake_rmsnorm_train_6554d4eb4362141cb058"),
            (262144, 720896, "cake_rmsnorm_train_420788e663a2c3a810f6"),
            (720896, 1572864, "cake_rmsnorm_train_420788e663a2c3a810f6"),
            (1572864, None, "cake_rmsnorm_train_420788e663a2c3a810f6"),
        ),
        "bwd_h512": (
            (0, 4, "cake_rmsnorm_train_64a299691c17079fe510"),
            (4, 16, "cake_rmsnorm_train_c167e0d26b363cf19e7b"),
            (16, 64, "cake_rmsnorm_train_bdfd77893d8f9db3c9bb"),
            (64, 1024, "cake_rmsnorm_train_09601d42628de24f6f87"),
            (1024, 4096, "cake_rmsnorm_train_8e7f4d811e55a351a22e"),
            (4096, 32768, "cake_rmsnorm_train_274b0055aace183f2367"),
            (32768, None, "cake_rmsnorm_train_cfaae3cec98f8501832b"),
        ),
        "bwd_w32_h512": (
            (0, 4, "cake_rmsnorm_train_8c261dfc6e34351d7a11"),
            (4, 16, "cake_rmsnorm_train_88e6252beaf7d056d555"),
            (16, 227, "cake_rmsnorm_train_2c60470d16e0a7fb6089"),
            (227, 1024, "cake_rmsnorm_train_9f07cea1d13b923ed91e"),
            (1024, 32768, "cake_rmsnorm_train_56ee8e66479625fd078a"),
            (32768, None, "cake_rmsnorm_train_a546f410ec8cf03ef20d"),
        ),
        "bwd_h1024": (
            (0, 4, "cake_rmsnorm_train_330930e9329d1e53aa5c"),
            (4, 64, "cake_rmsnorm_train_89fa3e964f6ac26dc6a2"),
            (64, 128, "cake_rmsnorm_train_5ad5f0962d5c5e8f63ac"),
            (128, 1024, "cake_rmsnorm_train_89fa3e964f6ac26dc6a2"),
            (1024, 1536, "cake_rmsnorm_train_5e32dd889d72a75e348b"),
            (1536, 2048, "cake_rmsnorm_train_33d9025920659f515c9f"),
            (2048, 32768, "cake_rmsnorm_train_af798d1034849cc692e9"),
            (32768, None, "cake_rmsnorm_train_acd331db1201bf2fb44e"),
        ),
        "bwd_w32_h1024": (
            (0, 4, "cake_rmsnorm_train_81e742d5d2cf1e0f790a"),
            (4, 8, "cake_rmsnorm_train_8501e95ea685c476cb50"),
            (8, 1024, "cake_rmsnorm_train_744d74299c2d45867c94"),
            (1024, 1536, "cake_rmsnorm_train_ba8feeec37d2bdde1fca"),
            (1536, 2048, "cake_rmsnorm_train_1a45ade5011422a0d019"),
            (2048, 4096, "cake_rmsnorm_train_1a45ade5011422a0d019"),
            (4096, 12288, "cake_rmsnorm_train_1a45ade5011422a0d019"),
            (12288, 32768, "cake_rmsnorm_train_1a45ade5011422a0d019"),
            (32768, 131072, "cake_rmsnorm_train_1a45ade5011422a0d019"),
            (131072, None, "cake_rmsnorm_train_2980f513dd6581ec98e7"),
        ),
        "bwd_h1280": (
            (0, 1, "cake_rmsnorm_train_8895b0224bb3d596a3a6"),
            (1, 128, "cake_rmsnorm_train_a5f2942d61e591f57a76"),
            (128, 512, "cake_rmsnorm_train_809235a3965d337a6877"),
            (512, 2048, "cake_rmsnorm_train_12bbedff93fcaafa475d"),
            (2048, None, "cake_rmsnorm_train_b2cf87e64c9399e38407"),
        ),
        "bwd_w32_h1280": (
            (0, 1, "cake_rmsnorm_train_103805d93149c8668264"),
            (1, 128, "cake_rmsnorm_train_8d8ef930b7d650c5538a"),
            (128, 512, "cake_rmsnorm_train_e476d0b2c0bd4285f16d"),
            (512, 4096, "cake_rmsnorm_train_80956d258172dfa76c2a"),
            (4096, None, "cake_rmsnorm_train_f6956a41e69eb126197e"),
        ),
        "bwd_h1536": (
            (0, 4, "cake_rmsnorm_train_db218c92c4c86b9da85a"),
            (4, 5, "cake_rmsnorm_train_c2a523de1ff578621a88"),
            (5, 64, "cake_rmsnorm_train_02e761bcf94053af6fc3"),
            (64, 214, "cake_rmsnorm_train_4aef8d1ff2c1c2af7b80"),
            (214, 1024, "cake_rmsnorm_train_7dc29629ff21625607f8"),
            (1024, 1216, "cake_rmsnorm_train_c1a9efdf06c3087b4efe"),
            (1216, 4096, "cake_rmsnorm_train_38e32d90f62f7cfc7b51"),
            (4096, 46592, "cake_rmsnorm_train_38e32d90f62f7cfc7b51"),
            (46592, 53184, "cake_rmsnorm_train_20d7a230ed8dc161a93a"),
            (53184, None, "cake_rmsnorm_train_38e32d90f62f7cfc7b51"),
        ),
        "bwd_w32_h1536": (
            (0, 4, "cake_rmsnorm_train_6d3986ee0dd962cafd21"),
            (4, 6, "cake_rmsnorm_train_744aa77460c886f4f429"),
            (6, 64, "cake_rmsnorm_train_166e288c9192be6b994c"),
            (64, 214, "cake_rmsnorm_train_698c3d36d67408394ccf"),
            (214, 1024, "cake_rmsnorm_train_baff9ecbafe7de1e6030"),
            (1024, 1216, "cake_rmsnorm_train_17d9cdfd157ba4b86c2b"),
            (1216, 4096, "cake_rmsnorm_train_e30c1899e859da4f4c68"),
            (4096, 46592, "cake_rmsnorm_train_e30c1899e859da4f4c68"),
            (46592, 53184, "cake_rmsnorm_train_51f46ff164b142ec8888"),
            (53184, None, "cake_rmsnorm_train_e30c1899e859da4f4c68"),
        ),
        "bwd_h2048": (
            (0, 4, "cake_rmsnorm_train_c5f62407b828a85f82ec"),
            (4, 16, "cake_rmsnorm_train_a951623c157d070e85ab"),
            (16, 63, "cake_rmsnorm_train_518cf16a7bf8e88042ee"),
            (63, 171, "cake_rmsnorm_train_ed36a8d998abba41f227"),
            (171, 16384, "cake_rmsnorm_train_cc26e316ae40d683a118"),
            (16384, 32768, "cake_rmsnorm_train_caf6978ed0a17fbf65b9"),
            (32768, 58496, "cake_rmsnorm_train_302f64db6abe79e5259e"),
            (58496, 61952, "cake_rmsnorm_train_302f64db6abe79e5259e"),
            (61952, 68160, "cake_rmsnorm_train_302f64db6abe79e5259e"),
            (68160, 131072, "cake_rmsnorm_train_87532fb038ad204517e1"),
            (131072, None, "cake_rmsnorm_train_33e5c382e06419d65390"),
        ),
        "bwd_w32_h2048": (
            (0, 4, "cake_rmsnorm_train_0017ab41f68d8689ebc2"),
            (4, 16, "cake_rmsnorm_train_65e5f2024d26a36f2e18"),
            (16, 64, "cake_rmsnorm_train_c2f08c957d4b0f17dddd"),
            (64, 171, "cake_rmsnorm_train_978741b43372951cb5b9"),
            (171, 16384, "cake_rmsnorm_train_c83c314c6dcf28ad7369"),
            (16384, 32768, "cake_rmsnorm_train_07f9f2fbea7163639c27"),
            (32768, 58496, "cake_rmsnorm_train_e98cb156abfe1666c76f"),
            (58496, 61952, "cake_rmsnorm_train_e98cb156abfe1666c76f"),
            (61952, 68160, "cake_rmsnorm_train_e98cb156abfe1666c76f"),
            (68160, 131072, "cake_rmsnorm_train_149b129cdd1468145b13"),
            (131072, None, "cake_rmsnorm_train_1166d499d4e846fcb2d6"),
        ),
        "bwd_h4096": (
            (0, 4, "cake_rmsnorm_train_945a0afdb82ab47eb643"),
            (4, 16, "cake_rmsnorm_train_41f44b4bc0b765350a50"),
            (16, 32, "cake_rmsnorm_train_dc7ed206fba49ab22d1d"),
            (32, 152, "cake_rmsnorm_train_12d53fe5b85b212fc800"),
            (152, 256, "cake_rmsnorm_train_50044ee3bd1ca54bd821"),
            (256, 1024, "cake_rmsnorm_train_ad2bb26ea5916baef963"),
            (1024, 20480, "cake_rmsnorm_train_5572b2671862fe670ef4"),
            (20480, 65536, "cake_rmsnorm_train_ef9f8d46c22e4af7aef1"),
            (65536, 262144, "cake_rmsnorm_train_f0aebb74650c8dc3f128"),
            (262144, None, "cake_rmsnorm_train_35888fb3b729e1f8e239"),
        ),
        "bwd_w32_h4096": (
            (0, 4, "cake_rmsnorm_train_f6bd1634a115bd281055"),
            (4, 16, "cake_rmsnorm_train_b378ba007691c582171e"),
            (16, 32, "cake_rmsnorm_train_8274f36f6fd2f50dd508"),
            (32, 152, "cake_rmsnorm_train_27615d0e36c9f2fe0856"),
            (152, 256, "cake_rmsnorm_train_3fd77c8ce97e8a523abb"),
            (256, 1024, "cake_rmsnorm_train_16545056022785f2a7d8"),
            (1024, 20480, "cake_rmsnorm_train_47a01388444960da7aae"),
            (20480, 48640, "cake_rmsnorm_train_626aed743358c27513f1"),
            (48640, 65536, "cake_rmsnorm_train_a8eb754739a1589de3ec"),
            (65536, 262144, "cake_rmsnorm_train_bd22da7e2d4914167050"),
            (262144, None, "cake_rmsnorm_train_36790f79b8845172d188"),
        ),
        "bwd_h5120": (
            (0, 4, "cake_rmsnorm_train_dbc680f356be4cd643de"),
            (4, 16, "cake_rmsnorm_train_29bcbde1d271ea989fa8"),
            (16, 64, "cake_rmsnorm_train_080e0a6c3b18e9bad790"),
            (64, 192, "cake_rmsnorm_train_2fc4932fbde0bc848b32"),
            (192, 1024, "cake_rmsnorm_train_bfde9b88e800fc34722c"),
            (1024, 36301, "cake_rmsnorm_train_bfde9b88e800fc34722c"),
            (36301, 61952, "cake_rmsnorm_train_2fc4932fbde0bc848b32"),
            (61952, 393216, "cake_rmsnorm_train_e21acade55e05f342053"),
            (393216, None, "cake_rmsnorm_train_826e6a14a93bd988be4d"),
        ),
        "bwd_w32_h5120": (
            (0, 4, "cake_rmsnorm_train_d47138228b123e55fe5d"),
            (4, 5, "cake_rmsnorm_train_05396cd3a077d3772fdc"),
            (5, 16, "cake_rmsnorm_train_cb6539573725ac861f94"),
            (16, 192, "cake_rmsnorm_train_139d162a8bb29d1b2d35"),
            (192, 1024, "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48"),
            (1024, 36301, "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48"),
            (36301, 61952, "cake_rmsnorm_train_79811a76205d7d7dd197"),
            (61952, 196608, "cake_rmsnorm_train_05078a174a09119cfe3c"),
            (196608, None, "cake_rmsnorm_train_91a40f3094182a35c784"),
        ),
        "bwd_h6144": (
            (0, 4, "cake_rmsnorm_train_e37752b056dd4e4a0bee"),
            (4, 6, "cake_rmsnorm_train_fdfb73350c1a926a4929"),
            (6, 16, "cake_rmsnorm_train_e6475d8fa504ef4d7d2d"),
            (16, 896, "cake_rmsnorm_train_44508df7dab2ab688f85"),
            (896, 4096, "cake_rmsnorm_train_61421bc72e57d14d8a32"),
            (4096, 24575, "cake_rmsnorm_train_322574af704b51aef0bb"),
            (24575, 65536, "cake_rmsnorm_train_455174e1a3be71c79452"),
            (65536, 131072, "cake_rmsnorm_train_850fba46397a84b8cd7b"),
            (131072, None, "cake_rmsnorm_train_38337af146209d6a60ea"),
        ),
        "bwd_residual_h6144": (
            (0, 4, "cake_rmsnorm_train_deeb187630473818923d"),
            (4, 6, "cake_rmsnorm_train_94c547eccfe2ca37ee0f"),
            (6, 31, "cake_rmsnorm_train_3a55646d574035bfa271"),
            (31, 32, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (32, 48, "cake_rmsnorm_train_3a55646d574035bfa271"),
            (48, 56, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (56, 304, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (304, 1920, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (1920, 131072, "cake_rmsnorm_train_aac3ebbf0e11c453c472"),
            (131072, None, "cake_rmsnorm_train_115dcc719a6de4b079f1"),
        ),
        "bwd_w32_h6144": (
            (0, 4, "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9"),
            (4, 12, "cake_rmsnorm_train_18b8b8054c7053942bcd"),
            (12, 16, "cake_rmsnorm_train_f4d41a0991647b82ac21"),
            (16, 1024, "cake_rmsnorm_train_b3e6531e46667901ab91"),
            (1024, 4096, "cake_rmsnorm_train_41b80b6af97974feb181"),
            (4096, 49152, "cake_rmsnorm_train_9a8c0396edd85d4d545e"),
            (49152, 83392, "cake_rmsnorm_train_9a8c0396edd85d4d545e"),
            (83392, 131072, "cake_rmsnorm_train_cb1aee125561ad8d4e29"),
            (131072, None, "cake_rmsnorm_train_5a635713b08645d57f48"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 4, "cake_rmsnorm_train_a6b1a6e565dc98b49e1a"),
            (4, 10, "cake_rmsnorm_train_c29a8f856283578103dc"),
            (10, 31, "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c"),
            (31, 32, "cake_rmsnorm_train_71d36617003ab7742a77"),
            (32, 48, "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c"),
            (48, 320, "cake_rmsnorm_train_71d36617003ab7742a77"),
            (320, 1824, "cake_rmsnorm_train_71d36617003ab7742a77"),
            (1824, 131072, "cake_rmsnorm_train_5fbcc18d24f37f0ba607"),
            (131072, None, "cake_rmsnorm_train_3c390339fa8c82311a0a"),
        ),
    },
    "sm_107a": {
        "fwd_h1024": (
            (0, 8, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (8, 16, "cake_rmsnorm_train_9e8a70075f28be558bba"),
            (16, 256, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (256, 1024, "cake_rmsnorm_train_9e8a70075f28be558bba"),
            (1024, 4096, "cake_rmsnorm_train_6e5ca1f45eefb79bee7e"),
            (4096, 8192, "cake_rmsnorm_train_9e8a70075f28be558bba"),
            (8192, 8447, "cake_rmsnorm_train_cffbc5c8de9c874b207c"),
            (8447, 13312, "cake_rmsnorm_train_547372db70c0ca0be362"),
            (13312, 32768, "cake_rmsnorm_train_9e8a70075f28be558bba"),
            (32768, None, "cake_rmsnorm_train_cffbc5c8de9c874b207c"),
        ),
        "fwd_w32_h1024": (
            (0, 8, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (8, 16, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (16, 512, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (512, 1536, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (1536, 4096, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (4096, 8192, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (8192, 12288, "cake_rmsnorm_train_f5110367098beea13130"),
            (12288, 16384, "cake_rmsnorm_train_364ed18ca31e5b754d4e"),
            (16384, 16895, "cake_rmsnorm_train_8bd2d83654da4780c48a"),
            (16895, 24576, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (24576, 32768, "cake_rmsnorm_train_d99009bfce4043a5dffe"),
            (32768, None, "cake_rmsnorm_train_008fb179743126fb2d35"),
        ),
        "fwd_h128": (
            (0, 32, "cake_rmsnorm_train_3808022dd402387eed16"),
            (32, 64, "cake_rmsnorm_train_628267fc724c2c8f0e07"),
            (64, 2048, "cake_rmsnorm_train_3808022dd402387eed16"),
            (2048, 4096, "cake_rmsnorm_train_7221de56ded18727adaf"),
            (4096, 24576, "cake_rmsnorm_train_3808022dd402387eed16"),
            (24576, 32768, "cake_rmsnorm_train_7221de56ded18727adaf"),
            (32768, None, "cake_rmsnorm_train_a05a2262fd5fa6912442"),
        ),
        "fwd_w32_h128": (
            (0, 32, "cake_rmsnorm_train_513f4ffbfefd9d019863"),
            (32, 64, "cake_rmsnorm_train_1ecf9c260e418bc70046"),
            (64, 2048, "cake_rmsnorm_train_513f4ffbfefd9d019863"),
            (2048, 4096, "cake_rmsnorm_train_04ea6484fdf4a9b14ee0"),
            (4096, 16384, "cake_rmsnorm_train_513f4ffbfefd9d019863"),
            (16384, 32768, "cake_rmsnorm_train_1ecf9c260e418bc70046"),
            (32768, None, "cake_rmsnorm_train_67e0dfc12fbae0354e51"),
        ),
        "fwd_h1280": (
            (0, 1, "cake_rmsnorm_train_03a74dfe62df95c2a752"),
            (1, 4, "cake_rmsnorm_train_4d9988d4680c7c3c02a5"),
            (4, 8, "cake_rmsnorm_train_03a74dfe62df95c2a752"),
            (8, 16, "cake_rmsnorm_train_007e82b24390c88156ac"),
            (16, 128, "cake_rmsnorm_train_4d9988d4680c7c3c02a5"),
            (128, 512, "cake_rmsnorm_train_03a74dfe62df95c2a752"),
            (512, 1024, "cake_rmsnorm_train_007e82b24390c88156ac"),
            (1024, 1536, "cake_rmsnorm_train_03b5516039c533b453b9"),
            (1536, 2048, "cake_rmsnorm_train_4d9988d4680c7c3c02a5"),
            (2048, 4096, "cake_rmsnorm_train_03a74dfe62df95c2a752"),
            (4096, 12288, "cake_rmsnorm_train_007e82b24390c88156ac"),
            (12288, 16384, "cake_rmsnorm_train_03b5516039c533b453b9"),
            (16384, None, "cake_rmsnorm_train_007e82b24390c88156ac"),
        ),
        "fwd_w32_h1280": (
            (0, 4, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (4, 8, "cake_rmsnorm_train_c43709cadf31ed04033c"),
            (8, 128, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (128, 512, "cake_rmsnorm_train_c43709cadf31ed04033c"),
            (512, 1024, "cake_rmsnorm_train_58c2fce1ea0b40758e91"),
            (1024, 1536, "cake_rmsnorm_train_62e8e2756e015c39bf5c"),
            (1536, 2048, "cake_rmsnorm_train_6258b6dac6840cc6bd58"),
            (2048, 8192, "cake_rmsnorm_train_62e8e2756e015c39bf5c"),
            (8192, 8703, "cake_rmsnorm_train_99b209a37f0203af57ce"),
            (8703, 9984, "cake_rmsnorm_train_51ca3c875efd4b1a5746"),
            (9984, 16384, "cake_rmsnorm_train_99b209a37f0203af57ce"),
            (16384, None, "cake_rmsnorm_train_58c2fce1ea0b40758e91"),
        ),
        "fwd_h1536": (
            (0, 2048, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (2048, 12288, "cake_rmsnorm_train_4cdc4c495b3f59ff5069"),
            (12288, None, "cake_rmsnorm_train_682675bda578f5b7a312"),
        ),
        "fwd_w32_h1536": (
            (0, 2048, "cake_rmsnorm_train_21d3416baa877808371e"),
            (2048, 12288, "cake_rmsnorm_train_92cf7b87f1ebffb1b12b"),
            (12288, None, "cake_rmsnorm_train_a30e5a3b69b7c183992f"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_b44107e05796c2922187"),
            (1024, 8192, "cake_rmsnorm_train_a4da85136fba4588239f"),
            (8192, 12288, "cake_rmsnorm_train_0ad4d53c4a0df501a99d"),
            (12288, 16384, "cake_rmsnorm_train_0ad4d53c4a0df501a99d"),
            (16384, None, "cake_rmsnorm_train_bde56bd5a68c493f07a2"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (1024, 8192, "cake_rmsnorm_train_a72c6ece1900f748eaa2"),
            (8192, 12288, "cake_rmsnorm_train_595f506004038a65cc63"),
            (12288, 16384, "cake_rmsnorm_train_40210950f2e7ee48bc3e"),
            (16384, None, "cake_rmsnorm_train_40210950f2e7ee48bc3e"),
        ),
        "fwd_h256": (
            (0, 6144, "cake_rmsnorm_train_c5ca409f44e169148301"),
            (6144, 32768, "cake_rmsnorm_train_bbb05314930d192dfe7b"),
            (32768, 98304, "cake_rmsnorm_train_466331ff3cd2d6c62738"),
            (98304, None, "cake_rmsnorm_train_37f663183cbff23c2893"),
        ),
        "fwd_w32_h256": (
            (0, 6144, "cake_rmsnorm_train_8f2756937eaac6b02502"),
            (6144, 32768, "cake_rmsnorm_train_a8fffd93682c5e382dc5"),
            (32768, 98304, "cake_rmsnorm_train_32bebe5a7782311d60a0"),
            (98304, None, "cake_rmsnorm_train_18e3c0a5204006b3a56f"),
        ),
        "fwd_h4096": (
            (0, 512, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (512, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 4096, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (4096, 20480, "cake_rmsnorm_train_bf933e1660cab8e5787b"),
            (20480, 49152, "cake_rmsnorm_train_bf933e1660cab8e5787b"),
            (49152, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 512, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (512, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 4096, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (4096, 20480, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (20480, 49152, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (49152, None, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
        ),
        "fwd_h512": (
            (0, 256, "cake_rmsnorm_train_7673978dc698790054d2"),
            (256, 6144, "cake_rmsnorm_train_2a42e5c02b240f8709b1"),
            (6144, None, "cake_rmsnorm_train_60f47c5d00570ecf7740"),
        ),
        "fwd_w32_h512": (
            (0, 256, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (256, 6144, "cake_rmsnorm_train_79fecc7016e44e940733"),
            (6144, None, "cake_rmsnorm_train_61accbcc02891615f286"),
        ),
        "fwd_h5120": (
            (0, 512, "cake_rmsnorm_train_fd1097c45c1e328871fd"),
            (512, 1152, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
            (1152, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, 20480, "cake_rmsnorm_train_044d4a2aa0553c16db06"),
            (20480, 49152, "cake_rmsnorm_train_044d4a2aa0553c16db06"),
            (49152, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 512, "cake_rmsnorm_train_ba28d2c0c2c81a2c714e"),
            (512, 1152, "cake_rmsnorm_train_1defffdf2712cae36784"),
            (1152, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, 20480, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
            (20480, 49152, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
            (49152, None, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, 4096, "cake_rmsnorm_train_cf1d05546951bdbd9f5d"),
            (4096, None, "cake_rmsnorm_train_e7f4221a7c26b9dfdd23"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 4096, "cake_rmsnorm_train_3d619cd1e3ae638cce3d"),
            (4096, None, "cake_rmsnorm_train_fdfa500c83d95df0f5c1"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_e646bbdbd852884953da"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_6325c474ad9955a319e9"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h128": (
            (0, 1024, "cake_rmsnorm_train_9346600882e53a589509"),
            (1024, 2047, "cake_rmsnorm_train_401a018c68bc082078b2"),
            (2047, 65536, "cake_rmsnorm_train_d197bcc34403e94ee2bd"),
            (65536, None, "cake_rmsnorm_train_8b0b86c827709e73ee26"),
        ),
        "bwd_w32_h128": (
            (0, 1024, "cake_rmsnorm_train_ab8a320c36b909e7bf1e"),
            (1024, 2047, "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea"),
            (2047, 65536, "cake_rmsnorm_train_e23083c7af388f9e2f50"),
            (65536, None, "cake_rmsnorm_train_3e751b6f0ebbd6be9481"),
        ),
        "bwd_h256": (
            (0, 4, "cake_rmsnorm_train_780c64c6f2c4615bfbcf"),
            (4, 24, "cake_rmsnorm_train_f2ccf7197a641c39acbf"),
            (24, 64, "cake_rmsnorm_train_9fb6b410e91f9d92becf"),
            (64, 786432, "cake_rmsnorm_train_a385ae1681a2c078344e"),
            (786432, None, "cake_rmsnorm_train_44ef314181f3f2e93293"),
        ),
        "bwd_w32_h256": (
            (0, 4, "cake_rmsnorm_train_67cb56412bd46304b711"),
            (4, 24, "cake_rmsnorm_train_aa745395d7542d8a8e3a"),
            (24, 64, "cake_rmsnorm_train_6a0114cb0b27a780a4c3"),
            (64, 786432, "cake_rmsnorm_train_420788e663a2c3a810f6"),
            (786432, None, "cake_rmsnorm_train_b71bbe9852c16fe03114"),
        ),
        "bwd_h512": (
            (0, 4, "cake_rmsnorm_train_64a299691c17079fe510"),
            (4, 16, "cake_rmsnorm_train_c167e0d26b363cf19e7b"),
            (16, 576, "cake_rmsnorm_train_49af1e75185ddd637b76"),
            (576, 1024, "cake_rmsnorm_train_3feed5026e1156d2fba2"),
            (1024, 4096, "cake_rmsnorm_train_3feed5026e1156d2fba2"),
            (4096, 16384, "cake_rmsnorm_train_dfaf378aa8ad93e68bbd"),
            (16384, 49152, "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3"),
            (49152, None, "cake_rmsnorm_train_7764c8a34009f3728312"),
        ),
        "bwd_w32_h512": (
            (0, 4, "cake_rmsnorm_train_8c261dfc6e34351d7a11"),
            (4, 16, "cake_rmsnorm_train_88e6252beaf7d056d555"),
            (16, 384, "cake_rmsnorm_train_8f29656c0018a749a588"),
            (384, 1024, "cake_rmsnorm_train_b841d64bbcaefff5ce7b"),
            (1024, 4096, "cake_rmsnorm_train_fd08c6f436eb6071fa22"),
            (4096, 16384, "cake_rmsnorm_train_383f709cd5c4fadd9436"),
            (16384, 49152, "cake_rmsnorm_train_03efa681eb38f59b00c2"),
            (49152, None, "cake_rmsnorm_train_5117a0f693a160485d76"),
        ),
        "bwd_h1024": (
            (0, 4, "cake_rmsnorm_train_330930e9329d1e53aa5c"),
            (4, 8, "cake_rmsnorm_train_28647b7ceb414d4fd02c"),
            (8, 1536, "cake_rmsnorm_train_89fa3e964f6ac26dc6a2"),
            (1536, 2048, "cake_rmsnorm_train_0c581a133624ff396ab4"),
            (2048, 4095, "cake_rmsnorm_train_5e32dd889d72a75e348b"),
            (4095, None, "cake_rmsnorm_train_4572a519afb83afc3743"),
        ),
        "bwd_w32_h1024": (
            (0, 4, "cake_rmsnorm_train_81e742d5d2cf1e0f790a"),
            (4, 16, "cake_rmsnorm_train_8501e95ea685c476cb50"),
            (16, 1536, "cake_rmsnorm_train_744d74299c2d45867c94"),
            (1536, 4095, "cake_rmsnorm_train_561bea88a25c2f245c29"),
            (4095, None, "cake_rmsnorm_train_64423f2da36401d9de39"),
        ),
        "bwd_h1280": (
            (0, 1, "cake_rmsnorm_train_8895b0224bb3d596a3a6"),
            (1, 128, "cake_rmsnorm_train_a5f2942d61e591f57a76"),
            (128, 512, "cake_rmsnorm_train_809235a3965d337a6877"),
            (512, 6143, "cake_rmsnorm_train_12bbedff93fcaafa475d"),
            (6143, 32767, "cake_rmsnorm_train_b1339aac23ed13dedce9"),
            (32767, None, "cake_rmsnorm_train_f8cb729b7ec5f34002ca"),
        ),
        "bwd_w32_h1280": (
            (0, 4, "cake_rmsnorm_train_7905241de01123c84d2d"),
            (4, 128, "cake_rmsnorm_train_8d8ef930b7d650c5538a"),
            (128, 512, "cake_rmsnorm_train_e476d0b2c0bd4285f16d"),
            (512, 4096, "cake_rmsnorm_train_80956d258172dfa76c2a"),
            (4096, 6143, "cake_rmsnorm_train_c66b8cbd5626e2c1f2b4"),
            (6143, 32767, "cake_rmsnorm_train_de2f75d431899520db87"),
            (32767, None, "cake_rmsnorm_train_c5386a35eeb7cfbd1658"),
        ),
        "bwd_h1536": (
            (0, 4, "cake_rmsnorm_train_db218c92c4c86b9da85a"),
            (4, 16, "cake_rmsnorm_train_c2a523de1ff578621a88"),
            (16, 320, "cake_rmsnorm_train_1207b05663ed2b4de14b"),
            (320, 1024, "cake_rmsnorm_train_7dc29629ff21625607f8"),
            (1024, 12288, "cake_rmsnorm_train_88510c609fd688aab969"),
            (12288, 16384, "cake_rmsnorm_train_6a428da2227972ae18ae"),
            (16384, 45056, "cake_rmsnorm_train_56d26996239835327501"),
            (45056, 65536, "cake_rmsnorm_train_411e270c0575d26a77f4"),
            (65536, None, "cake_rmsnorm_train_c93d0266065acad326fb"),
        ),
        "bwd_w32_h1536": (
            (0, 4, "cake_rmsnorm_train_6d3986ee0dd962cafd21"),
            (4, 16, "cake_rmsnorm_train_744aa77460c886f4f429"),
            (16, 320, "cake_rmsnorm_train_137ba5328176fb07b37d"),
            (320, 1024, "cake_rmsnorm_train_baff9ecbafe7de1e6030"),
            (1024, 16384, "cake_rmsnorm_train_7c0dd63f0b011049603e"),
            (16384, 45056, "cake_rmsnorm_train_de8e07d4cb312b06dbc4"),
            (45056, 65536, "cake_rmsnorm_train_64ffe19b041f5bfa34fb"),
            (65536, None, "cake_rmsnorm_train_c9dc9e43af546ff815d3"),
        ),
        "bwd_h2048": (
            (0, 4, "cake_rmsnorm_train_c5f62407b828a85f82ec"),
            (4, 16, "cake_rmsnorm_train_a951623c157d070e85ab"),
            (16, 31, "cake_rmsnorm_train_518cf16a7bf8e88042ee"),
            (31, 256, "cake_rmsnorm_train_6720b4a159be2b84b325"),
            (256, 1024, "cake_rmsnorm_train_845cdcd8145ada631461"),
            (1024, 16384, "cake_rmsnorm_train_cc26e316ae40d683a118"),
            (16384, 50176, "cake_rmsnorm_train_845cdcd8145ada631461"),
            (50176, 50688, "cake_rmsnorm_train_395fedebf0299f78a745"),
            (50688, 65536, "cake_rmsnorm_train_21ee6fc93139c5e0cd41"),
            (65536, None, "cake_rmsnorm_train_395fedebf0299f78a745"),
        ),
        "bwd_w32_h2048": (
            (0, 4, "cake_rmsnorm_train_0017ab41f68d8689ebc2"),
            (4, 16, "cake_rmsnorm_train_65e5f2024d26a36f2e18"),
            (16, 256, "cake_rmsnorm_train_23c900134dc889302136"),
            (256, 1024, "cake_rmsnorm_train_838a9a4054731f7942af"),
            (1024, 16384, "cake_rmsnorm_train_e90d94fcdd76f08fe027"),
            (16384, 50176, "cake_rmsnorm_train_838a9a4054731f7942af"),
            (50176, None, "cake_rmsnorm_train_e90d94fcdd76f08fe027"),
        ),
        "bwd_h4096": (
            (0, 4, "cake_rmsnorm_train_945a0afdb82ab47eb643"),
            (4, 16, "cake_rmsnorm_train_41f44b4bc0b765350a50"),
            (16, 256, "cake_rmsnorm_train_c1de7e41da4550cca0bc"),
            (256, 1024, "cake_rmsnorm_train_a647fccc3d9fa9134733"),
            (1024, 4096, "cake_rmsnorm_train_d41a514f5210e9adc39f"),
            (4096, 24064, "cake_rmsnorm_train_ee4a9053bb25b8d49fcf"),
            (24064, 24576, "cake_rmsnorm_train_a647fccc3d9fa9134733"),
            (24576, 65536, "cake_rmsnorm_train_23573e0c42a194a04876"),
            (65536, None, "cake_rmsnorm_train_f11a86167db4f8e9a5da"),
        ),
        "bwd_w32_h4096": (
            (0, 4, "cake_rmsnorm_train_f6bd1634a115bd281055"),
            (4, 16, "cake_rmsnorm_train_b378ba007691c582171e"),
            (16, 256, "cake_rmsnorm_train_30ec8bb68cf4fe0c817b"),
            (256, 1024, "cake_rmsnorm_train_dc439f8335327e482025"),
            (1024, 4096, "cake_rmsnorm_train_b08633208e60b8ce94fd"),
            (4096, 24064, "cake_rmsnorm_train_0f1859b9fe9e47668a1c"),
            (24064, 65536, "cake_rmsnorm_train_0ff440daa6f9f304e544"),
            (65536, None, "cake_rmsnorm_train_a1f537fc94bde175bb3c"),
        ),
        "bwd_h5120": (
            (0, 4, "cake_rmsnorm_train_dbc680f356be4cd643de"),
            (4, 12, "cake_rmsnorm_train_904b1ad45d6219b4f43c"),
            (12, 31, "cake_rmsnorm_train_29bcbde1d271ea989fa8"),
            (31, 256, "cake_rmsnorm_train_080e0a6c3b18e9bad790"),
            (256, 511, "cake_rmsnorm_train_838c077df3a6d35d55b7"),
            (511, 640, "cake_rmsnorm_train_080e0a6c3b18e9bad790"),
            (640, 895, "cake_rmsnorm_train_838c077df3a6d35d55b7"),
            (895, 896, "cake_rmsnorm_train_080e0a6c3b18e9bad790"),
            (896, 4096, "cake_rmsnorm_train_838c077df3a6d35d55b7"),
            (4096, 19968, "cake_rmsnorm_train_6816fb80e97380f1bcbc"),
            (19968, 49152, "cake_rmsnorm_train_b8b0913b40f035f43b76"),
            (49152, None, "cake_rmsnorm_train_670f4cafe8c34a429d30"),
        ),
        "bwd_w32_h5120": (
            (0, 4, "cake_rmsnorm_train_d47138228b123e55fe5d"),
            (4, 15, "cake_rmsnorm_train_05396cd3a077d3772fdc"),
            (15, 256, "cake_rmsnorm_train_139d162a8bb29d1b2d35"),
            (256, 511, "cake_rmsnorm_train_cd612de56c0985bb4caa"),
            (511, 640, "cake_rmsnorm_train_139d162a8bb29d1b2d35"),
            (640, 895, "cake_rmsnorm_train_cd612de56c0985bb4caa"),
            (895, 1280, "cake_rmsnorm_train_139d162a8bb29d1b2d35"),
            (1280, 4096, "cake_rmsnorm_train_cd612de56c0985bb4caa"),
            (4096, 19968, "cake_rmsnorm_train_468cc4d1b8128187b4b3"),
            (19968, 32767, "cake_rmsnorm_train_de0cf457cd240f9c6422"),
            (32767, 131072, "cake_rmsnorm_train_e7cb6972099b7e10c035"),
            (131072, None, "cake_rmsnorm_train_13bd0c5f56e37867eb26"),
        ),
        "bwd_h6144": (
            (0, 4, "cake_rmsnorm_train_e37752b056dd4e4a0bee"),
            (4, 16, "cake_rmsnorm_train_fdfb73350c1a926a4929"),
            (16, 960, "cake_rmsnorm_train_0dafdf094615d080fcc9"),
            (960, 1151, "cake_rmsnorm_train_5d292172df772adb17ae"),
            (1151, 1152, "cake_rmsnorm_train_0dafdf094615d080fcc9"),
            (1152, 19456, "cake_rmsnorm_train_5d292172df772adb17ae"),
            (19456, 30720, "cake_rmsnorm_train_9cc51adfe9826056658b"),
            (30720, 131072, "cake_rmsnorm_train_ffbcfda26d306410e01f"),
            (131072, None, "cake_rmsnorm_train_ddedaa3be7466a4faa97"),
        ),
        "bwd_residual_h6144": (
            (0, 4, "cake_rmsnorm_train_deeb187630473818923d"),
            (4, 15, "cake_rmsnorm_train_94c547eccfe2ca37ee0f"),
            (15, 320, "cake_rmsnorm_train_ee1be07a89a0482de45e"),
            (320, 447, "cake_rmsnorm_train_0dcdc563be377247044e"),
            (447, 576, "cake_rmsnorm_train_ee1be07a89a0482de45e"),
            (576, 3584, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (3584, 7168, "cake_rmsnorm_train_0dcdc563be377247044e"),
            (7168, 131072, "cake_rmsnorm_train_853d4447e18f114f5da5"),
            (131072, None, "cake_rmsnorm_train_87d2866c1ea3bcc08b13"),
        ),
        "bwd_w32_h6144": (
            (0, 4, "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9"),
            (4, 16, "cake_rmsnorm_train_18b8b8054c7053942bcd"),
            (16, 1664, "cake_rmsnorm_train_5cca2d0fa4553f992406"),
            (1664, 2304, "cake_rmsnorm_train_b3e6531e46667901ab91"),
            (2304, 19456, "cake_rmsnorm_train_e77387f14390ba486055"),
            (19456, 28672, "cake_rmsnorm_train_bdfba721a43e0d093668"),
            (28672, None, "cake_rmsnorm_train_e77387f14390ba486055"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 4, "cake_rmsnorm_train_a6b1a6e565dc98b49e1a"),
            (4, 16, "cake_rmsnorm_train_c29a8f856283578103dc"),
            (16, 768, "cake_rmsnorm_train_3bbd4f848b4847bff893"),
            (768, 3584, "cake_rmsnorm_train_71d36617003ab7742a77"),
            (3584, 7168, "cake_rmsnorm_train_56f988dde54db37f09d9"),
            (7168, 131072, "cake_rmsnorm_train_b394bc75f9df18f002f2"),
            (131072, None, "cake_rmsnorm_train_647589443d4915ca8fde"),
        ),
    },
}

# Per architecture and route: the upper edge of every window but the last and the measured
# class of the outputs across it: "same" (bitwise identical on the common rows), "ulp" (the
# forward's ulp differences of the notes), "unmeasured".
EDGES: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {
    "sm_100a": {
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
        "fwd_h2048": ((1024, "ulp"), (2048, "ulp"), (4096, "ulp")),
        "fwd_w32_h2048": ((1024, "ulp"), (2048, "ulp"), (4096, "ulp")),
        "fwd_h256": (
            (8192, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
        ),
        "fwd_w32_h256": (
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
        "fwd_h5120": ((640, "same"), (6144, "same")),
        "fwd_w32_h5120": ((640, "same"), (6144, "same"), (16384, "same")),
        "fwd_h6144": ((768, "same"), (1024, "same")),
        "fwd_w32_h6144": ((768, "same"), (1024, "same"), (6144, "same")),
        "fwd_residual_h6144": ((256, "same"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "same"), (1024, "ulp")),
        "bwd_h128": (
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (8192, "unmeasured"),
            (12288, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "bwd_w32_h128": (
            (1024, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (8192, "unmeasured"),
            (12288, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "bwd_h256": (
            (4, "unmeasured"),
            (24, "unmeasured"),
            (64, "unmeasured"),
            (20480, "same"),
            (92160, "same"),
            (524288, "same"),
        ),
        "bwd_w32_h256": (
            (4, "unmeasured"),
            (24, "unmeasured"),
            (64, "unmeasured"),
            (20480, "same"),
            (92160, "same"),
            (864240, "unmeasured"),
        ),
        "bwd_h512": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (227, "same"),
            (1024, "same"),
            (1280, "same"),
            (4096, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "bwd_w32_h512": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (227, "same"),
            (292, "same"),
            (960, "same"),
        ),
        "bwd_h1024": (
            (4, "unmeasured"),
            (8, "unmeasured"),
            (64, "unmeasured"),
            (128, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "bwd_w32_h1024": (
            (4, "unmeasured"),
            (8, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (4096, "unmeasured"),
            (12288, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "bwd_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (4096, "unmeasured"),
            (8192, "unmeasured"),
        ),
        "bwd_w32_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (4096, "unmeasured"),
        ),
        "bwd_h1536": (
            (2, "unmeasured"),
            (64, "same"),
            (384, "same"),
            (1024, "same"),
            (1184, "unmeasured"),
            (77824, "unmeasured"),
            (78784, "unmeasured"),
            (131072, "unmeasured"),
            (214016, "unmeasured"),
        ),
        "bwd_w32_h1536": (
            (2, "unmeasured"),
            (64, "same"),
            (384, "same"),
            (1024, "same"),
            (1184, "unmeasured"),
            (77824, "unmeasured"),
            (131072, "unmeasured"),
            (214016, "unmeasured"),
        ),
        "bwd_h2048": (
            (4, "unmeasured"),
            (12, "unmeasured"),
            (64, "same"),
            (336, "same"),
            (3072, "same"),
            (4096, "same"),
            (7168, "same"),
            (11264, "same"),
            (16384, "unmeasured"),
            (58880, "unmeasured"),
            (131072, "same"),
        ),
        "bwd_w32_h2048": (
            (4, "unmeasured"),
            (14, "unmeasured"),
            (64, "same"),
            (336, "same"),
            (1024, "same"),
            (4096, "same"),
            (15403, "same"),
            (45056, "same"),
            (58880, "unmeasured"),
            (65536, "unmeasured"),
            (131072, "same"),
        ),
        "bwd_h4096": (
            (4, "unmeasured"),
            (12, "unmeasured"),
            (64, "same"),
            (1152, "same"),
            (4096, "same"),
            (4608, "unmeasured"),
            (16384, "unmeasured"),
            (36864, "unmeasured"),
            (44032, "unmeasured"),
            (65536, "unmeasured"),
            (131072, "same"),
        ),
        "bwd_w32_h4096": (
            (4, "unmeasured"),
            (14, "unmeasured"),
            (64, "same"),
            (1792, "same"),
            (3584, "same"),
            (36864, "same"),
            (44032, "unmeasured"),
            (65536, "unmeasured"),
            (131072, "same"),
        ),
        "bwd_h5120": (
            (4, "unmeasured"),
            (64, "same"),
            (767, "unmeasured"),
            (36864, "unmeasured"),
            (56256, "same"),
            (131072, "unmeasured"),
        ),
        "bwd_w32_h5120": (
            (4, "unmeasured"),
            (64, "same"),
            (767, "unmeasured"),
            (36864, "unmeasured"),
            (56256, "unmeasured"),
            (131072, "unmeasured"),
        ),
        "bwd_h6144": (
            (4, "unmeasured"),
            (64, "same"),
            (1152, "same"),
            (4096, "same"),
            (42496, "unmeasured"),
            (65536, "same"),
            (131072, "same"),
        ),
        "bwd_residual_h6144": (
            (2, "unmeasured"),
            (64, "same"),
            (1408, "unmeasured"),
            (65536, "unmeasured"),
            (131072, "same"),
        ),
        "bwd_w32_h6144": (
            (4, "unmeasured"),
            (64, "same"),
            (1168, "same"),
            (3584, "same"),
            (65536, "same"),
            (131072, "same"),
        ),
        "bwd_w32_residual_h6144": (
            (4, "unmeasured"),
            (64, "same"),
            (1408, "same"),
            (65536, "same"),
            (131072, "same"),
        ),
    },
    "sm_103a": {
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
        "fwd_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_w32_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_w32_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_h256": (
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
        "fwd_h512": ((1024, "ulp"), (4096, "same"), (16384, "same")),
        "fwd_w32_h512": ((1024, "ulp"), (4096, "same"), (16384, "same")),
        "fwd_h5120": ((640, "same"), (6144, "same")),
        "fwd_w32_h5120": ((640, "same"), (6144, "same")),
        "fwd_h6144": ((1024, "same"),),
        "fwd_w32_h6144": ((1024, "same"), (6144, "same")),
        "fwd_residual_h6144": ((256, "same"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "same"), (1024, "ulp")),
        "bwd_h128": (
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (8192, "unmeasured"),
            (12288, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "bwd_w32_h128": (
            (1024, "unmeasured"),
            (2048, "unmeasured"),
            (8192, "unmeasured"),
            (12288, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "bwd_h256": (
            (4, "unmeasured"),
            (24, "unmeasured"),
            (64, "unmeasured"),
            (1080, "same"),
            (2048, "unmeasured"),
            (262144, "unmeasured"),
            (720896, "same"),
            (1572864, "same"),
        ),
        "bwd_w32_h256": (
            (4, "unmeasured"),
            (24, "unmeasured"),
            (64, "unmeasured"),
            (1080, "same"),
            (2048, "unmeasured"),
            (262144, "unmeasured"),
            (720896, "same"),
            (1572864, "same"),
        ),
        "bwd_h512": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (64, "same"),
            (1024, "same"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_w32_h512": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (227, "same"),
            (1024, "same"),
            (32768, "same"),
        ),
        "bwd_h1024": (
            (4, "unmeasured"),
            (64, "unmeasured"),
            (128, "unmeasured"),
            (1024, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (32768, "unmeasured"),
        ),
        "bwd_w32_h1024": (
            (4, "unmeasured"),
            (8, "unmeasured"),
            (1024, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (4096, "unmeasured"),
            (12288, "unmeasured"),
            (32768, "unmeasured"),
            (131072, "unmeasured"),
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
            (4, "unmeasured"),
            (5, "unmeasured"),
            (64, "same"),
            (214, "same"),
            (1024, "same"),
            (1216, "unmeasured"),
            (4096, "unmeasured"),
            (46592, "unmeasured"),
            (53184, "unmeasured"),
        ),
        "bwd_w32_h1536": (
            (4, "unmeasured"),
            (6, "unmeasured"),
            (64, "same"),
            (214, "same"),
            (1024, "same"),
            (1216, "unmeasured"),
            (4096, "unmeasured"),
            (46592, "unmeasured"),
            (53184, "unmeasured"),
        ),
        "bwd_h2048": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (63, "same"),
            (171, "unmeasured"),
            (16384, "unmeasured"),
            (32768, "same"),
            (58496, "same"),
            (61952, "same"),
            (68160, "same"),
            (131072, "same"),
        ),
        "bwd_w32_h2048": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (64, "same"),
            (171, "unmeasured"),
            (16384, "unmeasured"),
            (32768, "same"),
            (58496, "same"),
            (61952, "same"),
            (68160, "same"),
            (131072, "same"),
        ),
        "bwd_h4096": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (32, "same"),
            (152, "same"),
            (256, "same"),
            (1024, "unmeasured"),
            (20480, "unmeasured"),
            (65536, "unmeasured"),
            (262144, "unmeasured"),
        ),
        "bwd_w32_h4096": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (32, "same"),
            (152, "same"),
            (256, "same"),
            (1024, "unmeasured"),
            (20480, "unmeasured"),
            (48640, "unmeasured"),
            (65536, "unmeasured"),
            (262144, "same"),
        ),
        "bwd_h5120": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (64, "unmeasured"),
            (192, "unmeasured"),
            (1024, "unmeasured"),
            (36301, "unmeasured"),
            (61952, "same"),
            (393216, "same"),
        ),
        "bwd_w32_h5120": (
            (4, "unmeasured"),
            (5, "unmeasured"),
            (16, "unmeasured"),
            (192, "unmeasured"),
            (1024, "unmeasured"),
            (36301, "unmeasured"),
            (61952, "same"),
            (196608, "same"),
        ),
        "bwd_h6144": (
            (4, "unmeasured"),
            (6, "unmeasured"),
            (16, "unmeasured"),
            (896, "unmeasured"),
            (4096, "unmeasured"),
            (24575, "unmeasured"),
            (65536, "unmeasured"),
            (131072, "unmeasured"),
        ),
        "bwd_residual_h6144": (
            (4, "unmeasured"),
            (6, "unmeasured"),
            (31, "same"),
            (32, "same"),
            (48, "same"),
            (56, "unmeasured"),
            (304, "unmeasured"),
            (1920, "unmeasured"),
            (131072, "unmeasured"),
        ),
        "bwd_w32_h6144": (
            (4, "unmeasured"),
            (12, "unmeasured"),
            (16, "unmeasured"),
            (1024, "unmeasured"),
            (4096, "same"),
            (49152, "same"),
            (83392, "same"),
            (131072, "same"),
        ),
        "bwd_w32_residual_h6144": (
            (4, "unmeasured"),
            (10, "unmeasured"),
            (31, "same"),
            (32, "same"),
            (48, "unmeasured"),
            (320, "unmeasured"),
            (1824, "unmeasured"),
            (131072, "unmeasured"),
        ),
    },
    "sm_107a": {
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
        "bwd_h128": ((1024, "unmeasured"), (2047, "unmeasured"), (65536, "unmeasured")),
        "bwd_w32_h128": (
            (1024, "unmeasured"),
            (2047, "unmeasured"),
            (65536, "unmeasured"),
        ),
        "bwd_h256": (
            (4, "unmeasured"),
            (24, "unmeasured"),
            (64, "unmeasured"),
            (786432, "same"),
        ),
        "bwd_w32_h256": (
            (4, "unmeasured"),
            (24, "unmeasured"),
            (64, "unmeasured"),
            (786432, "same"),
        ),
        "bwd_h512": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (576, "same"),
            (1024, "same"),
            (4096, "same"),
            (16384, "same"),
            (49152, "same"),
        ),
        "bwd_w32_h512": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (384, "same"),
            (1024, "same"),
            (4096, "same"),
            (16384, "same"),
            (49152, "same"),
        ),
        "bwd_h1024": (
            (4, "unmeasured"),
            (8, "unmeasured"),
            (1536, "unmeasured"),
            (2048, "unmeasured"),
            (4095, "unmeasured"),
        ),
        "bwd_w32_h1024": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (1536, "unmeasured"),
            (4095, "unmeasured"),
        ),
        "bwd_h1280": (
            (1, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (6143, "unmeasured"),
            (32767, "unmeasured"),
        ),
        "bwd_w32_h1280": (
            (4, "unmeasured"),
            (128, "unmeasured"),
            (512, "unmeasured"),
            (4096, "unmeasured"),
            (6143, "unmeasured"),
            (32767, "unmeasured"),
        ),
        "bwd_h1536": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (320, "same"),
            (1024, "same"),
            (12288, "same"),
            (16384, "same"),
            (45056, "unmeasured"),
            (65536, "unmeasured"),
        ),
        "bwd_w32_h1536": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (320, "same"),
            (1024, "same"),
            (16384, "same"),
            (45056, "unmeasured"),
            (65536, "unmeasured"),
        ),
        "bwd_h2048": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (31, "same"),
            (256, "same"),
            (1024, "ulp"),
            (16384, "same"),
            (50176, "same"),
            (50688, "unmeasured"),
            (65536, "unmeasured"),
        ),
        "bwd_w32_h2048": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (256, "same"),
            (1024, "same"),
            (16384, "same"),
            (50176, "same"),
        ),
        "bwd_h4096": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (256, "same"),
            (1024, "same"),
            (4096, "same"),
            (24064, "unmeasured"),
            (24576, "unmeasured"),
            (65536, "unmeasured"),
        ),
        "bwd_w32_h4096": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (256, "same"),
            (1024, "same"),
            (4096, "same"),
            (24064, "unmeasured"),
            (65536, "unmeasured"),
        ),
        "bwd_h5120": (
            (4, "unmeasured"),
            (12, "unmeasured"),
            (31, "same"),
            (256, "same"),
            (511, "same"),
            (640, "same"),
            (895, "same"),
            (896, "same"),
            (4096, "same"),
            (19968, "same"),
            (49152, "same"),
        ),
        "bwd_w32_h5120": (
            (4, "unmeasured"),
            (15, "unmeasured"),
            (256, "same"),
            (511, "same"),
            (640, "same"),
            (895, "same"),
            (1280, "same"),
            (4096, "unmeasured"),
            (19968, "unmeasured"),
            (32767, "unmeasured"),
            (131072, "unmeasured"),
        ),
        "bwd_h6144": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (960, "same"),
            (1151, "same"),
            (1152, "same"),
            (19456, "same"),
            (30720, "unmeasured"),
            (131072, "unmeasured"),
        ),
        "bwd_residual_h6144": (
            (4, "unmeasured"),
            (15, "unmeasured"),
            (320, "same"),
            (447, "same"),
            (576, "same"),
            (3584, "same"),
            (7168, "unmeasured"),
            (131072, "unmeasured"),
        ),
        "bwd_w32_h6144": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (1664, "same"),
            (2304, "same"),
            (19456, "same"),
            (28672, "same"),
        ),
        "bwd_w32_residual_h6144": (
            (4, "unmeasured"),
            (16, "unmeasured"),
            (768, "same"),
            (3584, "same"),
            (7168, "unmeasured"),
            (131072, "unmeasured"),
        ),
    },
}

EDGE_CLASSES = ["same", "ulp", "dx", "unmeasured"]

# Per architecture: the backward routes that run the deterministic pair in every window, each
# with the measured reason; every other backward route ships the single kernel in at least one
# window (the exporter's ``DET_PAIR_ROUTES``; the registry gate test reads this table).
DET_PAIR_ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {},
    "sm_103a": {},
    "sm_107a": {},
}

# The shared-memory sizes (KB per CTA of a kernel resident on every SM) the co-residence of the
# backward windows is measured beside, and per architecture and route the windows measured:
# ``(above, at_most, KB)``, KB the smallest size beside which the window waits on any measured
# board of the architecture (None: it starts beside every measured size on every board); the
# exporter's ``SMEM_WAITS``, checked by the stall test (no board waits below KB).
SMEM_SIDE_KB = (100, 132)
SMEM_WAITS: dict[str, dict[str, tuple[tuple[int, int | None, int | None], ...]]] = {
    "sm_100a": {
        "bwd_h1024": (
            (0, 4, None),
            (4, 8, None),
            (8, 64, None),
            (64, 128, None),
            (128, 1024, None),
            (1024, 2048, None),
            (2048, 32768, None),
            (32768, None, None),
        ),
        "bwd_h128": (
            (0, 1024, None),
            (1024, 2048, None),
            (2048, 8192, None),
            (8192, 12288, None),
            (12288, 16384, None),
            (16384, None, None),
        ),
        "bwd_h1280": (
            (0, 1, None),
            (1, 128, None),
            (128, 512, None),
            (512, 4096, None),
            (4096, 8192, None),
            (8192, None, None),
        ),
        "bwd_h1536": (
            (0, 2, 100),
            (2, 64, None),
            (64, 384, None),
            (384, 1024, 132),
            (1024, 1184, 100),
            (1184, 77824, 132),
            (77824, 78784, 100),
            (78784, 131072, None),
            (131072, 214016, 132),
            (214016, None, None),
        ),
        "bwd_h2048": (
            (0, 4, 100),
            (4, 12, 100),
            (12, 64, None),
            (64, 336, None),
            (336, 3072, 100),
            (3072, 4096, 100),
            (4096, 7168, 100),
            (7168, 11264, 132),
            (11264, 16384, 100),
            (16384, 58880, 100),
            (58880, 131072, None),
            (131072, None, 132),
        ),
        "bwd_h256": (
            (0, 4, 100),
            (4, 24, 100),
            (24, 64, 100),
            (64, 20480, None),
            (20480, 92160, None),
            (92160, 524288, None),
            (524288, None, None),
        ),
        "bwd_h4096": (
            (0, 4, 100),
            (4, 12, 100),
            (12, 64, None),
            (64, 1152, None),
            (1152, 4096, 132),
            (4096, 4608, 132),
            (4608, 16384, 100),
            (16384, 36864, 100),
            (36864, 44032, 132),
            (44032, 65536, 100),
            (65536, 131072, None),
            (131072, None, None),
        ),
        "bwd_h512": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 227, None),
            (227, 1024, None),
            (1024, 1280, 132),
            (1280, 4096, None),
            (4096, 32768, 100),
            (32768, None, 100),
        ),
        "bwd_h5120": (
            (0, 4, 100),
            (4, 64, 100),
            (64, 767, None),
            (767, 36864, None),
            (36864, 56256, None),
            (56256, 131072, 132),
            (131072, None, None),
        ),
        "bwd_h6144": (
            (0, 4, 100),
            (4, 64, None),
            (64, 1152, 132),
            (1152, 4096, 132),
            (4096, 42496, 100),
            (42496, 65536, 100),
            (65536, 131072, 100),
            (131072, None, 100),
        ),
        "bwd_residual_h6144": (
            (0, 2, 100),
            (2, 64, None),
            (64, 1408, 100),
            (1408, 65536, 100),
            (65536, 131072, 100),
            (131072, None, 100),
        ),
        "bwd_w32_h1024": (
            (0, 4, None),
            (4, 8, None),
            (8, 1024, None),
            (1024, 2048, None),
            (2048, 4096, None),
            (4096, 12288, None),
            (12288, 32768, None),
            (32768, None, None),
        ),
        "bwd_w32_h128": (
            (0, 1024, None),
            (1024, 1536, None),
            (1536, 2048, None),
            (2048, 8192, None),
            (8192, 12288, None),
            (12288, 16384, None),
            (16384, None, None),
        ),
        "bwd_w32_h1280": (
            (0, 1, None),
            (1, 128, None),
            (128, 512, None),
            (512, 4096, None),
            (4096, None, None),
        ),
        "bwd_w32_h1536": (
            (0, 2, 100),
            (2, 64, None),
            (64, 384, None),
            (384, 1024, 132),
            (1024, 1184, None),
            (1184, 77824, 132),
            (77824, 131072, None),
            (131072, 214016, 132),
            (214016, None, None),
        ),
        "bwd_w32_h2048": (
            (0, 4, 100),
            (4, 14, 100),
            (14, 64, None),
            (64, 336, None),
            (336, 1024, None),
            (1024, 4096, 132),
            (4096, 15403, 100),
            (15403, 45056, 132),
            (45056, 58880, 132),
            (58880, 65536, 100),
            (65536, 131072, None),
            (131072, None, 132),
        ),
        "bwd_w32_h256": (
            (0, 4, 100),
            (4, 24, 100),
            (24, 64, 100),
            (64, 20480, None),
            (20480, 92160, None),
            (92160, 864240, None),
            (864240, None, None),
        ),
        "bwd_w32_h4096": (
            (0, 4, 100),
            (4, 14, 100),
            (14, 64, None),
            (64, 1792, None),
            (1792, 3584, None),
            (3584, 36864, None),
            (36864, 44032, 132),
            (44032, 65536, 100),
            (65536, 131072, None),
            (131072, None, None),
        ),
        "bwd_w32_h512": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 227, None),
            (227, 292, None),
            (292, 960, None),
            (960, None, None),
        ),
        "bwd_w32_h5120": (
            (0, 4, 100),
            (4, 64, 100),
            (64, 767, None),
            (767, 36864, None),
            (36864, 56256, 100),
            (56256, 131072, 132),
            (131072, None, None),
        ),
        "bwd_w32_h6144": (
            (0, 4, 100),
            (4, 64, None),
            (64, 1168, 132),
            (1168, 3584, 132),
            (3584, 65536, 100),
            (65536, 131072, 100),
            (131072, None, 100),
        ),
        "bwd_w32_residual_h6144": (
            (0, 4, 100),
            (4, 64, None),
            (64, 1408, 100),
            (1408, 65536, 100),
            (65536, 131072, 100),
            (131072, None, 100),
        ),
    },
    "sm_103a": {
        "bwd_h1024": (
            (0, 4, None),
            (4, 64, None),
            (64, 128, None),
            (128, 1024, None),
            (1024, 1536, None),
            (1536, 2048, None),
            (2048, 32768, None),
            (32768, None, None),
        ),
        "bwd_h128": (
            (0, 1024, None),
            (1024, 2048, None),
            (2048, 8192, None),
            (8192, 12288, None),
            (12288, 16384, None),
            (16384, None, None),
        ),
        "bwd_h1280": (
            (0, 1, None),
            (1, 128, None),
            (128, 512, None),
            (512, 2048, None),
            (2048, None, None),
        ),
        "bwd_h1536": (
            (0, 4, 100),
            (4, 5, 100),
            (5, 64, None),
            (64, 214, None),
            (214, 1024, 132),
            (1024, 1216, 100),
            (1216, 4096, 132),
            (4096, 46592, 132),
            (46592, 53184, None),
            (53184, None, 132),
        ),
        "bwd_h2048": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 63, None),
            (63, 171, None),
            (171, 16384, 132),
            (16384, 32768, 100),
            (32768, 58496, 100),
            (58496, 61952, 100),
            (61952, 68160, 100),
            (68160, 131072, None),
            (131072, None, 132),
        ),
        "bwd_h256": (
            (0, 4, 100),
            (4, 24, 100),
            (24, 64, 100),
            (64, 1080, None),
            (1080, 2048, None),
            (2048, 262144, None),
            (262144, 720896, None),
            (720896, 1572864, None),
            (1572864, None, None),
        ),
        "bwd_h4096": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 32, None),
            (32, 152, None),
            (152, 256, None),
            (256, 1024, None),
            (1024, 20480, None),
            (20480, 65536, 100),
            (65536, 262144, 100),
            (262144, None, None),
        ),
        "bwd_h512": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 64, None),
            (64, 1024, None),
            (1024, 4096, None),
            (4096, 32768, None),
            (32768, None, None),
        ),
        "bwd_h5120": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 64, 100),
            (64, 192, 132),
            (192, 1024, None),
            (1024, 36301, None),
            (36301, 61952, 132),
            (61952, 393216, 132),
            (393216, None, 132),
        ),
        "bwd_h6144": (
            (0, 4, 100),
            (4, 6, 100),
            (6, 16, None),
            (16, 896, 100),
            (896, 4096, 132),
            (4096, 24575, 132),
            (24575, 65536, 100),
            (65536, 131072, 100),
            (131072, None, 100),
        ),
        "bwd_residual_h6144": (
            (0, 4, 100),
            (4, 6, 100),
            (6, 31, None),
            (31, 32, 100),
            (32, 48, None),
            (48, 56, 100),
            (56, 304, 100),
            (304, 1920, 100),
            (1920, 131072, 100),
            (131072, None, 100),
        ),
        "bwd_w32_h1024": (
            (0, 4, None),
            (4, 8, None),
            (8, 1024, None),
            (1024, 1536, None),
            (1536, 2048, None),
            (2048, 4096, None),
            (4096, 12288, None),
            (12288, 32768, None),
            (32768, 131072, None),
            (131072, None, None),
        ),
        "bwd_w32_h128": (
            (0, 1024, None),
            (1024, 2048, None),
            (2048, 8192, None),
            (8192, 12288, None),
            (12288, 16384, None),
            (16384, None, None),
        ),
        "bwd_w32_h1280": (
            (0, 1, None),
            (1, 128, None),
            (128, 512, None),
            (512, 4096, None),
            (4096, None, None),
        ),
        "bwd_w32_h1536": (
            (0, 4, 100),
            (4, 6, 100),
            (6, 64, None),
            (64, 214, None),
            (214, 1024, 132),
            (1024, 1216, 100),
            (1216, 4096, 132),
            (4096, 46592, 132),
            (46592, 53184, 100),
            (53184, None, 132),
        ),
        "bwd_w32_h2048": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 64, None),
            (64, 171, None),
            (171, 16384, 132),
            (16384, 32768, 100),
            (32768, 58496, 100),
            (58496, 61952, 100),
            (61952, 68160, 100),
            (68160, 131072, None),
            (131072, None, 132),
        ),
        "bwd_w32_h256": (
            (0, 4, 100),
            (4, 24, 100),
            (24, 64, 100),
            (64, 1080, None),
            (1080, 2048, None),
            (2048, 262144, None),
            (262144, 720896, None),
            (720896, 1572864, None),
            (1572864, None, None),
        ),
        "bwd_w32_h4096": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 32, None),
            (32, 152, None),
            (152, 256, None),
            (256, 1024, None),
            (1024, 20480, None),
            (20480, 48640, 132),
            (48640, 65536, 100),
            (65536, 262144, None),
            (262144, None, None),
        ),
        "bwd_w32_h512": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 227, None),
            (227, 1024, None),
            (1024, 32768, None),
            (32768, None, None),
        ),
        "bwd_w32_h5120": (
            (0, 4, 100),
            (4, 5, 100),
            (5, 16, 100),
            (16, 192, 100),
            (192, 1024, None),
            (1024, 36301, None),
            (36301, 61952, 132),
            (61952, 196608, 132),
            (196608, None, 132),
        ),
        "bwd_w32_h6144": (
            (0, 4, 100),
            (4, 12, 100),
            (12, 16, None),
            (16, 1024, 100),
            (1024, 4096, 132),
            (4096, 49152, 132),
            (49152, 83392, 132),
            (83392, 131072, 100),
            (131072, None, 100),
        ),
        "bwd_w32_residual_h6144": (
            (0, 4, 100),
            (4, 10, 100),
            (10, 31, None),
            (31, 32, 100),
            (32, 48, None),
            (48, 320, 100),
            (320, 1824, 100),
            (1824, 131072, 100),
            (131072, None, 100),
        ),
    },
    "sm_107a": {
        "bwd_h1024": (
            (0, 4, None),
            (4, 8, None),
            (8, 1536, None),
            (1536, 2048, None),
            (2048, 4095, None),
            (4095, None, None),
        ),
        "bwd_h128": (
            (0, 1024, None),
            (1024, 2047, None),
            (2047, 65536, None),
            (65536, None, None),
        ),
        "bwd_h1280": (
            (0, 1, None),
            (1, 128, None),
            (128, 512, None),
            (512, 6143, None),
            (6143, 32767, None),
            (32767, None, None),
        ),
        "bwd_h1536": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 320, None),
            (320, 1024, None),
            (1024, 12288, 100),
            (12288, 16384, 100),
            (16384, 45056, 100),
            (45056, 65536, 100),
            (65536, None, None),
        ),
        "bwd_h2048": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 31, None),
            (31, 256, None),
            (256, 1024, 100),
            (1024, 16384, 100),
            (16384, 50176, 100),
            (50176, 50688, None),
            (50688, 65536, 100),
            (65536, None, None),
        ),
        "bwd_h256": (
            (0, 4, 100),
            (4, 24, 100),
            (24, 64, 100),
            (64, 786432, None),
            (786432, None, None),
        ),
        "bwd_h4096": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 256, None),
            (256, 1024, 100),
            (1024, 4096, 100),
            (4096, 24064, 100),
            (24064, 24576, 100),
            (24576, 65536, 100),
            (65536, None, 132),
        ),
        "bwd_h512": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 576, None),
            (576, 1024, None),
            (1024, 4096, 100),
            (4096, 16384, 100),
            (16384, 49152, 100),
            (49152, None, None),
        ),
        "bwd_h5120": (
            (0, 4, 100),
            (4, 12, 100),
            (12, 31, 100),
            (31, 256, None),
            (256, 511, 100),
            (511, 640, None),
            (640, 895, 100),
            (895, 896, None),
            (896, 4096, 100),
            (4096, 19968, 100),
            (19968, 49152, 100),
            (49152, None, 132),
        ),
        "bwd_h6144": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 960, 132),
            (960, 1151, 100),
            (1151, 1152, 132),
            (1152, 19456, 100),
            (19456, 30720, 100),
            (30720, 131072, 100),
            (131072, None, 100),
        ),
        "bwd_residual_h6144": (
            (0, 4, 100),
            (4, 15, 100),
            (15, 320, 100),
            (320, 447, 100),
            (447, 576, 100),
            (576, 3584, 100),
            (3584, 7168, 100),
            (7168, 131072, 100),
            (131072, None, 100),
        ),
        "bwd_w32_h1024": (
            (0, 4, None),
            (4, 16, None),
            (16, 1536, None),
            (1536, 4095, None),
            (4095, None, None),
        ),
        "bwd_w32_h128": (
            (0, 1024, None),
            (1024, 2047, None),
            (2047, 65536, None),
            (65536, None, None),
        ),
        "bwd_w32_h1280": (
            (0, 4, None),
            (4, 128, None),
            (128, 512, None),
            (512, 4096, None),
            (4096, 6143, None),
            (6143, 32767, None),
            (32767, None, None),
        ),
        "bwd_w32_h1536": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 320, None),
            (320, 1024, None),
            (1024, 16384, None),
            (16384, 45056, 100),
            (45056, 65536, 100),
            (65536, None, None),
        ),
        "bwd_w32_h2048": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 256, None),
            (256, 1024, 100),
            (1024, 16384, None),
            (16384, 50176, 100),
            (50176, None, None),
        ),
        "bwd_w32_h256": (
            (0, 4, 100),
            (4, 24, 100),
            (24, 64, 100),
            (64, 786432, None),
            (786432, None, None),
        ),
        "bwd_w32_h4096": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 256, None),
            (256, 1024, 100),
            (1024, 4096, 100),
            (4096, 24064, 100),
            (24064, 65536, 100),
            (65536, None, 132),
        ),
        "bwd_w32_h512": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 384, None),
            (384, 1024, None),
            (1024, 4096, None),
            (4096, 16384, 100),
            (16384, 49152, 100),
            (49152, None, None),
        ),
        "bwd_w32_h5120": (
            (0, 4, 100),
            (4, 15, 100),
            (15, 256, None),
            (256, 511, 100),
            (511, 640, None),
            (640, 895, 100),
            (895, 1280, None),
            (1280, 4096, 100),
            (4096, 19968, 100),
            (19968, 32767, 100),
            (32767, 131072, 100),
            (131072, None, 132),
        ),
        "bwd_w32_h6144": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 1664, 132),
            (1664, 2304, 132),
            (2304, 19456, 100),
            (19456, 28672, 100),
            (28672, None, 100),
        ),
        "bwd_w32_residual_h6144": (
            (0, 4, 100),
            (4, 16, 100),
            (16, 768, 100),
            (768, 3584, 100),
            (3584, 7168, 100),
            (7168, 131072, 100),
            (131072, None, 100),
        ),
    },
}

# Per architecture and route: the token-count ranges ``(above, at_most)`` whose program needs a
# whole SM and waits beside any kernel resident on every SM, whatever its carveout: the
# deterministic package's whole-SM programs in this package's det-pair windows and forward
# (the exporter's ``DET_WHOLE_SM``; no single-kernel window); ``WHOLE_SM_REMEDY`` the
# deterministic facade's remedy.  The stall tests count these as not measured (det-inherited).
WHOLE_SM: dict[str, dict[str, tuple[tuple[int, int | None], ...]]] = {
    "sm_107a": {
        "fwd_h6144": ((4096, None),),
        "fwd_w32_h6144": ((4096, None),),
        "bwd_h1536": ((12288, 65536),),
        "bwd_w32_h1536": ((16384, 65536),),
        "bwd_h2048": ((256, 1024), (16384, 50176), (50688, 65536)),
        "bwd_w32_h2048": ((256, 1024), (16384, 50176)),
        "bwd_h4096": ((1024, 24064),),
        "bwd_w32_h4096": ((1024, 65536),),
        "bwd_residual_h6144": ((320, 447), (3584, 7168)),
        "bwd_w32_residual_h6144": ((3584, 7168),),
    },
}
WHOLE_SM_REMEDY = "Beside those, leave some SMs free of the resident kernel."

NOTES = """
Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window ``above <
T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows; the table
below lists them).  The forward programs are the deterministic package's, with its
windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``, and at a
forward edge between programs of different per-row thread groups ``rstd`` moves by at
most two FP32 ulps (three on R200 at the residual H = 6144 edge 1024) and ``y`` by at
most one BF16 ulp, on up to about 32 % of the rows (measured on B200, GB300 and R200).
A backward window runs one of three forms.  Up to T 16 every backward route, and up to
64 rows ``bwd_h256`` and ``bwd_w32_h256``, run the one-cluster kernel, marked ``(one
cluster)``: one launch of 1 to 8 CTAs forming one thread-block cluster over every row,
with no workspace and no atomic, one program up to T 4 and one above (up to 4, 24 and 64
rows on the H = 256 routes).  Lower one-cluster tops, where a larger sampled token count
ran the backward, the forward and backward or the training step slower than the program
above: on sm_100a up to T 14 on ``bwd_w32_h2048`` and ``bwd_w32_h4096``, up to T 12 on
``bwd_h2048`` and ``bwd_h4096``, up to T 8 on ``bwd_h1024`` and ``bwd_w32_h1024``, up
to T 4 on ``bwd_h5120``, ``bwd_h6144``, ``bwd_w32_h5120``, ``bwd_w32_h6144`` and
``bwd_w32_residual_h6144`` and up to T 2 on ``bwd_h1536``, ``bwd_residual_h6144`` and
``bwd_w32_h1536``; on sm_103a up to T 12 on ``bwd_w32_h6144``, up to T 10 on
``bwd_w32_residual_h6144``, up to T 8 on ``bwd_w32_h1024``, up to T 6 on ``bwd_h6144``,
``bwd_residual_h6144`` and ``bwd_w32_h1536``, up to T 5 on ``bwd_h1536`` and
``bwd_w32_h5120`` and up to T 4 on ``bwd_h1024`` and ``bwd_h5120``; on sm_107a up to
T 15 on ``bwd_residual_h6144`` and ``bwd_w32_h5120``, up to T 12 on ``bwd_h5120``, up to
T 8 on ``bwd_h1024`` and up to T 4 on ``bwd_w32_h1280``.  No one-cluster window runs on
``bwd_h128``, ``bwd_w32_h128`` and ``bwd_h1280``, nor on sm_100a and sm_103a on
``bwd_w32_h1280``.  It forms
``dx`` with the single kernel's formula (its row sums in another order, so ``dx`` may
differ from the single kernel's by rounding) and ``dw`` in FP32 in a fixed order: the
same bits on every launch of a token count.  Where the notes below name token counts at
or below those tops, they describe the windows above them.  The single kernel computes
``dx`` row by row (bitwise reproducible) and adds the
``dw`` sums of its CTAs into FP32 column sums (FP32 adds) in a scheduling-dependent
order: its ``dw`` is not bitwise reproducible, even for a fixed ``T``.  The top window
of ``bwd_h256`` on sm_100a and sm_107a, on sm_103a of ``bwd_h2048``, ``bwd_h4096``,
``bwd_h5120``, ``bwd_h6144`` and ``bwd_residual_h6144`` and on sm_107a of
``bwd_h1536``, ``bwd_h2048``, ``bwd_h6144`` and ``bwd_residual_h6144`` (and of their
FP32-weight routes), the windows above T 49152 of sm_107a ``bwd_h512`` and above T 65536
of sm_107a ``bwd_h4096``, on sm_100a the windows above T 131072 of ``bwd_h2048``,
``bwd_h4096``, ``bwd_h6144`` and ``bwd_residual_h6144`` and above T 214016 of
``bwd_h1536`` (both weight dtypes), the lowest window of each of those routes that
runs its top window's program, the lowest window on sm_103a of ``bwd_h4096`` and
``bwd_w32_h4096``,
the window of ``bwd_w32_h512`` from T 227 to 292 on sm_100a, the sm_100a ``bwd_h256``
window from 20480 to 92160 rows, on sm_100a the windows of ``bwd_h4096`` from T 4608 to
16384 and from T 16384 to 36864 and of ``bwd_residual_h6144`` from T 1408 to 65536, run
a single kernel whose threads sum their ``dw`` columns in FP32 over every row they take
and add each sum to the FP32 sums once; its ``dw`` is not bitwise reproducible either.
In those sm_100a windows it is a faster variant of the single kernel of the route's top
window with the same ``dw`` arithmetic: on ``bwd_h4096`` with ``evict_last`` loads
(without read-ahead up to T 16384), on ``bwd_residual_h6144`` with 3 pipeline stages.
On sm_103a the window below the top window of ``bwd_w32_h4096``, ``bwd_h5120`` and
``bwd_w32_h5120``, on sm_107a the top window of ``bwd_h5120`` and ``bwd_w32_h5120``, on
sm_103a also the window from T 46592 to 53184 of ``bwd_h1536``, from T 68160 to 131072
of ``bwd_h2048`` and ``bwd_w32_h2048``, from T 83392 to 131072 of ``bwd_w32_h6144``,
and on sm_100a the windows from T 78784 to 131072 of ``bwd_h1536``, from T 77824 to
131072 of ``bwd_w32_h1536``, from T 65536 to 131072 of ``bwd_h2048``,
``bwd_w32_h2048``, ``bwd_h4096``, ``bwd_w32_h4096``, ``bwd_h6144``, ``bwd_w32_h6144``,
``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` and from T 56256 to 131072 of
``bwd_h5120`` and ``bwd_w32_h5120`` run a flush single kernel: each thread adds the
``dw`` products of a block of 8 rows in FP32 and adds each block sum to its FP32 column
sums (on sm_103a its windows end at T 262144 on ``bwd_w32_h4096``, T 196608 on
``bwd_w32_h5120`` and T 393216 on ``bwd_h5120``); its ``dw`` is not bitwise reproducible
either.  On sm_103a ``bwd_residual_h6144`` from T 1920 to 131072,
``bwd_w32_residual_h6144`` from T 1824 to 131072, ``bwd_h6144`` from T 24575 to 131072
and ``bwd_h4096`` from T 20480 to 262144, and on sm_107a ``bwd_h6144`` from T 30720 to
131072, ``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` from T 7168 to 131072,
``bwd_h4096`` from T 24576 to 65536 and ``bwd_w32_h5120`` from T 32767 to 131072 run a
part-row single kernel: each thread sums its ``dw`` columns in FP32 (on sm_107a
``bwd_w32_h5120`` in 8-row blocks), each CTA writes its sums as one FP32 row of a
scratch area of the workspace, and once every row is in, its helper CTAs (128; 96 on
sm_103a ``bwd_h6144`` up to T 65536, 256 on sm_107a ``bwd_h4096``) add the rows in FP32
and store ``dw``, with no atomic add of a sum; its ``dw`` is not bitwise reproducible
either (the row order follows scheduling).  On sm_103a from T 20480 to 65536 on
``bwd_h4096``, from T 24575 to 65536 on ``bwd_h6144`` and in every sm_107a window its
build cuts the chunk table for more slots per SM than the grid has CTAs (8 and 6 per SM
on sm_103a; 4 per SM on sm_107a, 8 on ``bwd_h4096``): more, shorter chunks, the grid
unchanged.  A window marked ``(det pair)`` runs the deterministic package's main kernel
and fixed-order dw reduce, rendered into this package; on ``bwd_h5120`` (both weight
dtypes) the det pair windows above T 36864 on sm_100a, 36301 on sm_103a and 19968 on
sm_107a run the deterministic package's programs of the rows above T 36864 (sm_100a),
36544 (sm_103a) and 19968 (sm_107a) up to T 56256 on sm_100a, up to T 61952 on sm_103a
and up to T 49152 on sm_107a (T 32767 on ``bwd_w32_h5120``), a single kernel above (on
sm_100a up to T 131072, the one-row-per-stage det pair above).  On sm_100a and sm_103a
``bwd_h1536``, ``bwd_w32_h1536``, ``bwd_h5120`` and ``bwd_w32_h5120`` run the
deterministic package's one-row-per-stage det pairs (each CTA one slot, rows strided
over the slots, eight SMs left out) from T 1184 to 77824 and from T 131072 to 214016
(H = 1536) and from T 767 to 36864 and above T 131072 (H = 5120) on sm_100a, and from
T 4096 to 46592 and above T 53184 (H = 1536) and from T 1024 to 36301 (H = 5120) on
sm_103a; below those sm_103a windows, from T 1216 to 4096 (H = 1536) and from T 192 to
1024 (H = 5120), they run on every SM: in records on B200 and GB200 (sm_100a) and on
GB300 (sm_103a) no sampled T in those windows had a backward call, a forward and
backward call or a checkpointed step slower than with the programs they replace.
The deterministic package's one-row-per-stage det pairs also run with that package's
slot plan (eight SMs left out) on sm_103a ``bwd_h2048`` and ``bwd_w32_h2048`` from
T 171 to 16384, ``bwd_h6144`` from T 896 to 4096 and ``bwd_h4096`` and
``bwd_w32_h4096`` from T 1024 to 20480.  On ``bwd_h128`` and ``bwd_w32_h128`` the
deterministic package's ``rpw`` det pair (each CTA one slot, 16-row tiles strided over
the slots) runs from T 2048 to 8192 on sm_100a and sm_103a (eight SMs left out) and
from T 2047 to 65536 on sm_107a (every SM).  On ``bwd_h1024`` the deterministic
package's one-row-per-stage det pairs (every SM) also run above T 32768 on sm_100a and
sm_103a, and on ``bwd_w32_h1024`` above T 32768 on sm_100a.  On sm_107a the
deterministic package's grouped det pairs (five-row tiles strided over the slots, one
SM left out) run on ``bwd_h1024`` and ``bwd_w32_h1024`` above T 4095 and on
``bwd_h1280`` and ``bwd_w32_h1280`` above T 6143 (three-row tiles above T 32767).
On sm_103a ``bwd_h256`` (both weight dtypes) every window above 64
rows runs a det pair.  Their main kernels are those the deterministic package runs for
those rows on a device with 152 or more SMs, except from 1080 to 2048 rows, where it is
the main kernel that package runs from 2048 to 5120 rows on sm_100a; their fixed-order
dw reduce is that package's, except from 720896 to 1572864 rows, where it is another of
that package's dw reduces.  On sm_103a ``bwd_h256`` and ``bwd_w32_h256`` run the
deterministic package's one-row-per-stage det pair (each CTA one slot, eight-row tiles
strided over the slots, eight SMs left out) from 2048 to 262144 rows.  On ``bwd_h256``
(both weight dtypes) the det pair the deterministic package runs above 262144 rows on
sm_100a also runs from 92160 to 524288 rows on sm_100a, and with FP32 weight up to
864240 rows.  The single kernel of the top window runs above.  On sm_103a
``bwd_w32_h512`` runs from T
1024 to 32768 the det pair the deterministic package runs from T 4096.  On sm_100a
``bwd_h512`` above T 4096, ``bwd_h2048`` from T 13608 to 58880, ``bwd_w32_h2048`` from T
58880 to 65536 (the FP32-weight det pair) and ``bwd_h6144`` from T 5632 to 42496 run the
det pair the deterministic package runs there, and ``bwd_h2048`` from T 58880 to 65536
the flush single kernel of its window above.  On sm_100a ``bwd_h1536`` from T 77824 to
78784 runs the det pair the deterministic package runs above T 161792, ``bwd_h4096``
from T 44032 to 65536 the det pair that package runs there, and ``bwd_w32_h4096`` from
T 44032 to 65536 the FP32-weight det pair that package runs there.  On sm_103a
``bwd_w32_h4096``
from T 20480 to 65536 runs FP32-weight det pairs of the deterministic package, split at T
48640 (above it the twin of that package's 512-thread row-issuer program).  On sm_100a
``bwd_w32_h5120`` from T 36864 to 56256 runs the FP32-pair det pair, which that package
runs at every T above 36864; above T 56256 the route runs the flush single kernel (to T
131072), which sums in plain FP32 words, and the one-row-per-stage det pair above.  On
sm_107a ``bwd_h4096`` from T 24064 to 24576 and ``bwd_w32_h5120``
from T 4096
to 32767 run the det pair the deterministic package runs there, and ``bwd_w32_h4096``
from T 24064 to 65536 the FP32-weight det pair the deterministic package's
``bwd_w32_h4096`` runs from T 24064 (that package cuts it at T 50176).  On sm_107a
``bwd_h1536`` and ``bwd_w32_h1536`` from T 45056, the deterministic package's top edge,
to 65536 run the
det pair that package runs there (its four-row block form).  On sm_107a ``bwd_h2048``
from T 50688 to 65536 runs the det pair the deterministic package runs there (its
eight-row block form), for speed.  On sm_100a ``bwd_w32_h1536`` from T 1024 to 1184,
``bwd_w32_h2048`` from T 1024 to 4096 and ``bwd_w32_h4096`` from T 1792 to 3584 run the
single kernel of their top window and ``bwd_w32_h2048`` from T 336 to 1024 a column-task
kernel (its CTAs first sum whole columns of ``dw`` over every row); on sm_107a
``bwd_w32_h6144`` from T 19456 to 28672 runs the det pair the deterministic package runs
above T 19456, ``bwd_w32_h2048`` from T 1024 to 16384 the single kernel of its largest
token counts, and ``bwd_w32_h1536`` from T 1024 to 16384 and ``bwd_w32_h6144`` from T
2304 to 19456 the single kernel of their top
window.  On sm_100a ``bwd_w32_h2048`` from T 64 to 336 and on sm_107a ``bwd_w32_h5120``
from T 896 to 1280 run the deterministic package's column-task program of those token
counts, and on sm_103a ``bwd_w32_h512`` from T 960 to 1024 the FP32-weight det pair the
deterministic package runs there: in a same-node comparison on B200, GB300 and VR200
(CUDA-graph span, L2 hot and flushed before every call) each took 0.54x to 0.97x the
time of the program it replaces at every sampled token
count.  On sm_100a ``bwd_h2048`` from T 45056 to 58880
runs the single kernel of its window from T 7168 to 11264 and ``bwd_w32_h2048`` from T
45056 to 58880 the single kernel of its neighbouring windows, within 2 % of the det
pair's time.  On sm_107a ``bwd_h6144`` from T 19456 to 30720 runs the det pair the
deterministic package runs there.  On sm_100a ``bwd_h512`` and ``bwd_w32_h512`` up to T
227 run the deterministic package's column-task program of that window, which is faster
there.  On sm_103a ``bwd_h512`` up to T 64, ``bwd_w32_h512`` up to T 227, ``bwd_h1536``
and ``bwd_w32_h1536`` from T 64 to 214, ``bwd_h2048`` from T 63 and ``bwd_w32_h2048``
from T 64 to 171, ``bwd_h4096`` and ``bwd_w32_h4096`` from T 32 to 152 and T 32 of
``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` run the deterministic package's
column-task programs of those token counts, and ``bwd_w32_h512`` from T 227 to 292 and
``bwd_w32_h1536`` from T 214 to 320 the det pair of the window above, which are faster
there.  On sm_103a ``bwd_residual_h6144`` from T 56 to 304 and
``bwd_w32_residual_h6144`` from T 48 to 320 run the deterministic package's column-task
program of those token counts (and its FP32-weight twin), which that package runs from T
0 to 1216: on GB300 it took 5.82, 8.13 and 9.34 us at T 64, 256 and 304 where the
whole-SM det pair took 7.65, 10.24 and 10.53 us.  On sm_103a ``bwd_h5120`` from T 17 to
64, ``bwd_w32_h5120`` from T 17 to 192, ``bwd_h6144`` from T 17 to 896 and
``bwd_w32_h6144`` from T 17 to 1024 run the deterministic package's column-task program
of the route in its ticket form (the form whose ``dx`` and ``dw`` the fixed-claim form
reproduces bit for bit): on GB300 (benchmark, CUDA-graph span, L2 flushed before every
call; one or two runs per T) no sampled T up to those token counts had a backward call,
a forward and backward call or a checkpointed step (two forward calls and a backward
call) slower than with the programs it replaces beyond the run-to-run band in all its
runs.  Above them the det pair of the window above runs: at T 96 on ``bwd_h5120``, T 256
on ``bwd_w32_h5120`` and T 1024 on ``bwd_h6144`` (and at some larger sampled T) a
forward and backward call or a checkpointed step with the ticket form was slower in both
runs, though its backward call alone was faster there.  On sm_107a ``bwd_h512`` up to T
576, ``bwd_w32_h512`` up to T 384, ``bwd_h1536`` from T 15 to 320, ``bwd_w32_h1536``
from T 1 to 320, ``bwd_h2048`` from T 31 and ``bwd_w32_h2048`` up to T 256,
``bwd_h4096`` from T 1 and ``bwd_w32_h4096`` up to T 256, ``bwd_h5120`` from T 31 to
256, from T 511 to 640 and T 896, ``bwd_w32_h5120`` from T 15 to 256, from T 511 to 640
and T 896, ``bwd_h6144`` from T 15 to 960 and T 1152, ``bwd_w32_h6144`` from T 1 to
1664, ``bwd_residual_h6144`` from T 15 to 320 and from T 447 to 576 and
``bwd_w32_residual_h6144`` up to T 768 run the deterministic package's column-task
programs of those token counts, which took 0.53x to 0.97x the time of the programs they
replace at every sampled T in two R200 records with 212 SMs (the slower record per T)
and were also faster there in a VR200 screen.  On sm_107a ``bwd_h256`` and
``bwd_w32_h256`` up to 786432 rows run the deterministic package's det pair of those
rows, which took 0.76x to 0.95x the time of the program it replaces at every sampled row
count in two R200 records with 212 SMs and was also faster in a VR200 screen; above
786432 rows the single kernel of the top window runs.  On 9 backward routes the lowest
window, from T 1 up to a token count from 1 to 31, runs a single kernel.  On
``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` (and their FP32-weight routes) the window above the one-cluster
windows, from T 17 up to T 64 (on sm_103a up to T 63 on ``bwd_h2048``, up to T 32 on
``bwd_h4096`` and ``bwd_w32_h4096``, up to T 48 on ``bwd_residual_h6144`` and
``bwd_w32_residual_h6144``, except T 32 of those two, and not on ``bwd_h5120``,
``bwd_w32_h5120``, ``bwd_h6144`` and ``bwd_w32_h6144``; on sm_107a only on ``bwd_h2048``
and ``bwd_h5120``, up to T 31), runs the fixed-claim form of the column-task kernel:
each CTA takes its rows and its column tasks by its index instead of by claim tickets,
so no CTA waits for another, and its ``dx`` and ``dw`` are bitwise those of the ticket
form.  Every program sums in plain FP32 words, except the FP32-pair det pairs above,
which add each block of rows into an FP32 pair (a sum and its compensation word).  The
tests hold no tolerance: they judge every floating output by its error against an FP64
reference beside the error of the naive PyTorch implementation (the same ops in BF16 on
the same BF16 inputs), over 32 independent random inputs of each case: in the L2 norm
and in the max norm the median and the largest error at most 1.005 times the naive's,
and 0 where the naive's is 0.
"""

FORWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window ``above <
T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows; the table
below lists them).  The forward programs are the deterministic package's, with its
windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``, and at a
forward edge between programs of different per-row thread groups ``rstd`` moves by at
most two FP32 ulps (three on R200 at the residual H = 6144 edge 1024) and ``y`` by at
most one BF16 ulp, on up to about 32 % of the rows (measured on B200, GB300 and R200).
A backward window runs one of three forms.  Up to T 16 every backward route, and up to
64 rows ``bwd_h256`` and ``bwd_w32_h256``, run the one-cluster kernel, marked ``(one
cluster)``: one launch of 1 to 8 CTAs forming one thread-block cluster over every row,
with no workspace and no atomic, one program up to T 4 and one above (up to 4, 24 and 64
rows on the H = 256 routes).  Lower one-cluster tops, where a larger sampled token count
ran the backward, the forward and backward or the training step slower than the program
above: on sm_100a up to T 14 on ``bwd_w32_h2048`` and ``bwd_w32_h4096``, up to T 12 on
``bwd_h2048`` and ``bwd_h4096``, up to T 8 on ``bwd_h1024`` and ``bwd_w32_h1024``, up
to T 4 on ``bwd_h5120``, ``bwd_h6144``, ``bwd_w32_h5120``, ``bwd_w32_h6144`` and
``bwd_w32_residual_h6144`` and up to T 2 on ``bwd_h1536``, ``bwd_residual_h6144`` and
``bwd_w32_h1536``; on sm_103a up to T 12 on ``bwd_w32_h6144``, up to T 10 on
``bwd_w32_residual_h6144``, up to T 8 on ``bwd_w32_h1024``, up to T 6 on ``bwd_h6144``,
``bwd_residual_h6144`` and ``bwd_w32_h1536``, up to T 5 on ``bwd_h1536`` and
``bwd_w32_h5120`` and up to T 4 on ``bwd_h1024`` and ``bwd_h5120``; on sm_107a up to
T 15 on ``bwd_residual_h6144`` and ``bwd_w32_h5120``, up to T 12 on ``bwd_h5120``, up to
T 8 on ``bwd_h1024`` and up to T 4 on ``bwd_w32_h1280``.  No one-cluster window runs on
``bwd_h128``, ``bwd_w32_h128`` and ``bwd_h1280``, nor on sm_100a and sm_103a on
``bwd_w32_h1280``.  It forms
``dx`` with the single kernel's formula (its row sums in another order, so ``dx`` may
differ from the single kernel's by rounding) and ``dw`` in FP32 in a fixed order: the
same bits on every launch of a token count.  Where the notes below name token counts at
or below those tops, they describe the windows above them.  The single kernel computes
``dx`` row by row (bitwise reproducible) and adds the
``dw`` sums of its CTAs into FP32 column sums (FP32 adds) in a scheduling-dependent
order: its ``dw`` is not bitwise reproducible, even for a fixed ``T``.  The top window
of ``bwd_h256`` on sm_100a and sm_107a, on sm_103a of ``bwd_h2048``, ``bwd_h4096``,
``bwd_h5120``, ``bwd_h6144`` and ``bwd_residual_h6144`` and on sm_107a of
``bwd_h1536``, ``bwd_h2048``, ``bwd_h6144`` and ``bwd_residual_h6144`` (and of their
FP32-weight routes), the windows above T 49152 of sm_107a ``bwd_h512`` and above T 65536
of sm_107a ``bwd_h4096``, on sm_100a the windows above T 131072 of ``bwd_h2048``,
``bwd_h4096``, ``bwd_h6144`` and ``bwd_residual_h6144`` and above T 214016 of
``bwd_h1536`` (both weight dtypes), the lowest window of each of those routes that
runs its top window's program, the lowest window on sm_103a of ``bwd_h4096`` and
``bwd_w32_h4096``,
the window of ``bwd_w32_h512`` from T 227 to 292 on sm_100a, the sm_100a ``bwd_h256``
window from 20480 to 92160 rows, on sm_100a the windows of ``bwd_h4096`` from T 4608 to
16384 and from T 16384 to 36864 and of ``bwd_residual_h6144`` from T 1408 to 65536, run
a single kernel whose threads sum their ``dw`` columns in FP32 over every row they take
and add each sum to the FP32 sums once; its ``dw`` is not bitwise reproducible either.
In those sm_100a windows it is a faster variant of the single kernel of the route's top
window with the same ``dw`` arithmetic: on ``bwd_h4096`` with ``evict_last`` loads
(without read-ahead up to T 16384), on ``bwd_residual_h6144`` with 3 pipeline stages.
On sm_103a the window below the top window of ``bwd_w32_h4096``, ``bwd_h5120`` and
``bwd_w32_h5120``, on sm_107a the top window of ``bwd_h5120`` and ``bwd_w32_h5120``, on
sm_103a also the window from T 46592 to 53184 of ``bwd_h1536``, from T 68160 to 131072
of ``bwd_h2048`` and ``bwd_w32_h2048``, from T 83392 to 131072 of ``bwd_w32_h6144``,
and on sm_100a the windows from T 78784 to 131072 of ``bwd_h1536``, from T 77824 to
131072 of ``bwd_w32_h1536``, from T 65536 to 131072 of ``bwd_h2048``,
``bwd_w32_h2048``, ``bwd_h4096``, ``bwd_w32_h4096``, ``bwd_h6144``, ``bwd_w32_h6144``,
``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` and from T 56256 to 131072 of
``bwd_h5120`` and ``bwd_w32_h5120`` run a flush single kernel: each thread adds the
``dw`` products of a block of 8 rows in FP32 and adds each block sum to its FP32 column
sums (on sm_103a its windows end at T 262144 on ``bwd_w32_h4096``, T 196608 on
``bwd_w32_h5120`` and T 393216 on ``bwd_h5120``); its ``dw`` is not bitwise reproducible
either.  On sm_103a ``bwd_residual_h6144`` from T 1920 to 131072,
``bwd_w32_residual_h6144`` from T 1824 to 131072, ``bwd_h6144`` from T 24575 to 131072
and ``bwd_h4096`` from T 20480 to 262144, and on sm_107a ``bwd_h6144`` from T 30720 to
131072, ``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` from T 7168 to 131072,
``bwd_h4096`` from T 24576 to 65536 and ``bwd_w32_h5120`` from T 32767 to 131072 run a
part-row single kernel: each thread sums its ``dw`` columns in FP32 (on sm_107a
``bwd_w32_h5120`` in 8-row blocks), each CTA writes its sums as one FP32 row of a
scratch area of the workspace, and once every row is in, its helper CTAs (128; 96 on
sm_103a ``bwd_h6144`` up to T 65536, 256 on sm_107a ``bwd_h4096``) add the rows in FP32
and store ``dw``, with no atomic add of a sum; its ``dw`` is not bitwise reproducible
either (the row order follows scheduling).  On sm_103a from T 20480 to 65536 on
``bwd_h4096``, from T 24575 to 65536 on ``bwd_h6144`` and in every sm_107a window its
build cuts the chunk table for more slots per SM than the grid has CTAs (8 and 6 per SM
on sm_103a; 4 per SM on sm_107a, 8 on ``bwd_h4096``): more, shorter chunks, the grid
unchanged.  A window marked ``(det pair)`` runs the deterministic package's main kernel
and fixed-order dw reduce, rendered into this package; on ``bwd_h5120`` (both weight
dtypes) the det pair windows above T 36864 on sm_100a, 36301 on sm_103a and 19968 on
sm_107a run the deterministic package's programs of the rows above T 36864 (sm_100a),
36544 (sm_103a) and 19968 (sm_107a) up to T 56256 on sm_100a, up to T 61952 on sm_103a
and up to T 49152 on sm_107a (T 32767 on ``bwd_w32_h5120``), a single kernel above (on
sm_100a up to T 131072, the one-row-per-stage det pair above).  On sm_100a and sm_103a
``bwd_h1536``, ``bwd_w32_h1536``, ``bwd_h5120`` and ``bwd_w32_h5120`` run the
deterministic package's one-row-per-stage det pairs (each CTA one slot, rows strided
over the slots, eight SMs left out) from T 1184 to 77824 and from T 131072 to 214016
(H = 1536) and from T 767 to 36864 and above T 131072 (H = 5120) on sm_100a, and from
T 4096 to 46592 and above T 53184 (H = 1536) and from T 1024 to 36301 (H = 5120) on
sm_103a; below those sm_103a windows, from T 1216 to 4096 (H = 1536) and from T 192 to
1024 (H = 5120), they run on every SM: in records on B200 and GB200 (sm_100a) and on
GB300 (sm_103a) no sampled T in those windows had a backward call, a forward and
backward call or a checkpointed step slower than with the programs they replace.
The deterministic package's one-row-per-stage det pairs also run with that package's
slot plan (eight SMs left out) on sm_103a ``bwd_h2048`` and ``bwd_w32_h2048`` from
T 171 to 16384, ``bwd_h6144`` from T 896 to 4096 and ``bwd_h4096`` and
``bwd_w32_h4096`` from T 1024 to 20480.  On ``bwd_h128`` and ``bwd_w32_h128`` the
deterministic package's ``rpw`` det pair (each CTA one slot, 16-row tiles strided over
the slots) runs from T 2048 to 8192 on sm_100a and sm_103a (eight SMs left out) and
from T 2047 to 65536 on sm_107a (every SM).  On ``bwd_h1024`` the deterministic
package's one-row-per-stage det pairs (every SM) also run above T 32768 on sm_100a and
sm_103a, and on ``bwd_w32_h1024`` above T 32768 on sm_100a.  On sm_107a the
deterministic package's grouped det pairs (five-row tiles strided over the slots, one
SM left out) run on ``bwd_h1024`` and ``bwd_w32_h1024`` above T 4095 and on
``bwd_h1280`` and ``bwd_w32_h1280`` above T 6143 (three-row tiles above T 32767).
On sm_103a ``bwd_h256`` (both weight dtypes) every window above 64
rows runs a det pair.  Their main kernels are those the deterministic package runs for
those rows on a device with 152 or more SMs, except from 1080 to 2048 rows, where it is
the main kernel that package runs from 2048 to 5120 rows on sm_100a; their fixed-order
dw reduce is that package's, except from 720896 to 1572864 rows, where it is another of
that package's dw reduces.  On sm_103a ``bwd_h256`` and ``bwd_w32_h256`` run the
deterministic package's one-row-per-stage det pair (each CTA one slot, eight-row tiles
strided over the slots, eight SMs left out) from 2048 to 262144 rows.  On ``bwd_h256``
(both weight dtypes) the det pair the deterministic package runs above 262144 rows on
sm_100a also runs from 92160 to 524288 rows on sm_100a, and with FP32 weight up to
864240 rows.  The single kernel of the top window runs above.  On sm_103a
``bwd_w32_h512`` runs from T
1024 to 32768 the det pair the deterministic package runs from T 4096.  On sm_100a
``bwd_h512`` above T 4096, ``bwd_h2048`` from T 13608 to 58880, ``bwd_w32_h2048`` from T
58880 to 65536 (the FP32-weight det pair) and ``bwd_h6144`` from T 5632 to 42496 run the
det pair the deterministic package runs there, and ``bwd_h2048`` from T 58880 to 65536
the flush single kernel of its window above.  On sm_100a ``bwd_h1536`` from T 77824 to
78784 runs the det pair the deterministic package runs above T 161792, ``bwd_h4096``
from T 44032 to 65536 the det pair that package runs there, and ``bwd_w32_h4096`` from
T 44032 to 65536 the FP32-weight det pair that package runs there.  On sm_103a
``bwd_w32_h4096``
from T 20480 to 65536 runs FP32-weight det pairs of the deterministic package, split at T
48640 (above it the twin of that package's 512-thread row-issuer program).  On sm_100a
``bwd_w32_h5120`` from T 36864 to 56256 runs the FP32-pair det pair, which that package
runs at every T above 36864; above T 56256 the route runs the flush single kernel (to T
131072), which sums in plain FP32 words, and the one-row-per-stage det pair above.  On
sm_107a ``bwd_h4096`` from T 24064 to 24576 and ``bwd_w32_h5120``
from T 4096
to 32767 run the det pair the deterministic package runs there, and ``bwd_w32_h4096``
from T 24064 to 65536 the FP32-weight det pair the deterministic package's
``bwd_w32_h4096`` runs from T 24064 (that package cuts it at T 50176).  On sm_107a
``bwd_h1536`` and ``bwd_w32_h1536`` from T 45056, the deterministic package's top edge,
to 65536 run the
det pair that package runs there (its four-row block form).  On sm_107a ``bwd_h2048``
from T 50688 to 65536 runs the det pair the deterministic package runs there (its
eight-row block form), for speed.  On sm_100a ``bwd_w32_h1536`` from T 1024 to 1184,
``bwd_w32_h2048`` from T 1024 to 4096 and ``bwd_w32_h4096`` from T 1792 to 3584 run the
single kernel of their top window and ``bwd_w32_h2048`` from T 336 to 1024 a column-task
kernel (its CTAs first sum whole columns of ``dw`` over every row); on sm_107a
``bwd_w32_h6144`` from T 19456 to 28672 runs the det pair the deterministic package runs
above T 19456, ``bwd_w32_h2048`` from T 1024 to 16384 the single kernel of its largest
token counts, and ``bwd_w32_h1536`` from T 1024 to 16384 and ``bwd_w32_h6144`` from T
2304 to 19456 the single kernel of their top
window.  On sm_100a ``bwd_w32_h2048`` from T 64 to 336 and on sm_107a ``bwd_w32_h5120``
from T 896 to 1280 run the deterministic package's column-task program of those token
counts, and on sm_103a ``bwd_w32_h512`` from T 960 to 1024 the FP32-weight det pair the
deterministic package runs there: in a same-node comparison on B200, GB300 and VR200
(CUDA-graph span, L2 hot and flushed before every call) each took 0.54x to 0.97x the
time of the program it replaces at every sampled token
count.  On sm_100a ``bwd_h2048`` from T 45056 to 58880
runs the single kernel of its window from T 7168 to 11264 and ``bwd_w32_h2048`` from T
45056 to 58880 the single kernel of its neighbouring windows, within 2 % of the det
pair's time.  On sm_107a ``bwd_h6144`` from T 19456 to 30720 runs the det pair the
deterministic package runs there.  On sm_100a ``bwd_h512`` and ``bwd_w32_h512`` up to T
227 run the deterministic package's column-task program of that window, which is faster
there.  On sm_103a ``bwd_h512`` up to T 64, ``bwd_w32_h512`` up to T 227, ``bwd_h1536``
and ``bwd_w32_h1536`` from T 64 to 214, ``bwd_h2048`` from T 63 and ``bwd_w32_h2048``
from T 64 to 171, ``bwd_h4096`` and ``bwd_w32_h4096`` from T 32 to 152 and T 32 of
``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` run the deterministic package's
column-task programs of those token counts, and ``bwd_w32_h512`` from T 227 to 292 and
``bwd_w32_h1536`` from T 214 to 320 the det pair of the window above, which are faster
there.  On sm_103a ``bwd_residual_h6144`` from T 56 to 304 and
``bwd_w32_residual_h6144`` from T 48 to 320 run the deterministic package's column-task
program of those token counts (and its FP32-weight twin), which that package runs from T
0 to 1216: on GB300 it took 5.82, 8.13 and 9.34 us at T 64, 256 and 304 where the
whole-SM det pair took 7.65, 10.24 and 10.53 us.  On sm_103a ``bwd_h5120`` from T 17 to
64, ``bwd_w32_h5120`` from T 17 to 192, ``bwd_h6144`` from T 17 to 896 and
``bwd_w32_h6144`` from T 17 to 1024 run the deterministic package's column-task program
of the route in its ticket form (the form whose ``dx`` and ``dw`` the fixed-claim form
reproduces bit for bit): on GB300 (benchmark, CUDA-graph span, L2 flushed before every
call; one or two runs per T) no sampled T up to those token counts had a backward call,
a forward and backward call or a checkpointed step (two forward calls and a backward
call) slower than with the programs it replaces beyond the run-to-run band in all its
runs.  Above them the det pair of the window above runs: at T 96 on ``bwd_h5120``, T 256
on ``bwd_w32_h5120`` and T 1024 on ``bwd_h6144`` (and at some larger sampled T) a
forward and backward call or a checkpointed step with the ticket form was slower in both
runs, though its backward call alone was faster there.  On sm_107a ``bwd_h512`` up to T
576, ``bwd_w32_h512`` up to T 384, ``bwd_h1536`` from T 15 to 320, ``bwd_w32_h1536``
from T 1 to 320, ``bwd_h2048`` from T 31 and ``bwd_w32_h2048`` up to T 256,
``bwd_h4096`` from T 1 and ``bwd_w32_h4096`` up to T 256, ``bwd_h5120`` from T 31 to
256, from T 511 to 640 and T 896, ``bwd_w32_h5120`` from T 15 to 256, from T 511 to 640
and T 896, ``bwd_h6144`` from T 15 to 960 and T 1152, ``bwd_w32_h6144`` from T 1 to
1664, ``bwd_residual_h6144`` from T 15 to 320 and from T 447 to 576 and
``bwd_w32_residual_h6144`` up to T 768 run the deterministic package's column-task
programs of those token counts, which took 0.53x to 0.97x the time of the programs they
replace at every sampled T in two R200 records with 212 SMs (the slower record per T)
and were also faster there in a VR200 screen.  On sm_107a ``bwd_h256`` and
``bwd_w32_h256`` up to 786432 rows run the deterministic package's det pair of those
rows, which took 0.76x to 0.95x the time of the program it replaces at every sampled row
count in two R200 records with 212 SMs and was also faster in a VR200 screen; above
786432 rows the single kernel of the top window runs.  On 9 backward routes the lowest
window, from T 1 up to a token count from 1 to 31, runs a single kernel.  On
``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` (and their FP32-weight routes) the window above the one-cluster
windows, from T 17 up to T 64 (on sm_103a up to T 63 on ``bwd_h2048``, up to T 32 on
``bwd_h4096`` and ``bwd_w32_h4096``, up to T 48 on ``bwd_residual_h6144`` and
``bwd_w32_residual_h6144``, except T 32 of those two, and not on ``bwd_h5120``,
``bwd_w32_h5120``, ``bwd_h6144`` and ``bwd_w32_h6144``; on sm_107a only on ``bwd_h2048``
and ``bwd_h5120``, up to T 31), runs the fixed-claim form of the column-task kernel:
each CTA takes its rows and its column tasks by its index instead of by claim tickets,
so no CTA waits for another, and its ``dx`` and ``dw`` are bitwise those of the ticket
form.  Every program sums in plain FP32 words, except the FP32-pair det pairs above,
which add each block of rows into an FP32 pair (a sum and its compensation word).  The
tests hold no tolerance: they judge every floating output by its error against an FP64
reference beside the error of the naive PyTorch implementation (the same ops in BF16 on
the same BF16 inputs), over 32 independent random inputs of each case: in the L2 norm
and in the max norm the median and the largest error at most 1.005 times the naive's,
and 0 where the naive's is 0.

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge: ``=`` bitwise-identical outputs on the common rows across it, ``~``
the forward's ulp differences (see above), ``?`` not yet measured.

The FP32-weight routes (``fwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows and are not listed; a route re-cut for the FP32
weight (another twin in a window) is listed with its own windows.

Programs that need a whole SM wait beside any kernel resident on every SM, whatever
its carveout: the deterministic package's whole-SM programs where this package runs
them (``WHOLE_SM``: det-pair windows and forward programs; no single-kernel window).
Beside those, leave some SMs free of the resident kernel.

* ``sm_107a`` ``fwd_h6144``: (4096, inf)
* ``sm_107a`` ``fwd_w32_h6144``: (4096, inf)

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
* ``sm_100a`` ``fwd_h2048``: (0, 1024] b44107e0 ~ (1024, 2048] 780b9833 ~
  (2048, 4096] b44107e0 ~ (4096, inf) 780b9833
* ``sm_100a`` ``fwd_h256``: (0, 8192] 0489c784 = (8192, 40960] 1b284bae =
  (40960, 81920] 86d51eff = (81920, 114688] 1388ab77 = (114688, 196608] 466331ff =
  (196608, 327680] ed9483d8 = (327680, inf) b8f32dd4
* ``sm_100a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_100a`` ``fwd_h512``: (0, 1024] 8757f6da ~ (1024, 4096] 7673978d =
  (4096, 24576] 3d20ab4a = (24576, inf) 3430e5d9
* ``sm_100a`` ``fwd_h5120``: (0, 640] 5eef28cd = (640, 6144] b5e173f5 =
  (6144, inf) 5eef28cd
* ``sm_100a`` ``fwd_w32_h5120``: (0, 640] ba28d2c0 = (640, 6144] 9ecb62c2 =
  (6144, 16384] b49de9b4 = (16384, inf) 9ecb62c2
* ``sm_100a`` ``fwd_h6144``: (0, 768] bad4db91 = (768, 1024] bad4db91 =
  (1024, inf) 288d7b1d
* ``sm_100a`` ``fwd_w32_h6144``: (0, 768] 2763cd5e = (768, 1024] 2763cd5e =
  (1024, 6144] 3d619cd1 = (6144, inf) 5f89e5fd
* ``sm_100a`` ``fwd_residual_h6144``: (0, 256] 308da182 = (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
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
* ``sm_103a`` ``fwd_h1536``: (0, 1536] 1ea88523 ~ (1536, 3584] 2fdbab2a =
  (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_103a`` ``fwd_h2048``: (0, 1024] c5a8099e = (1024, 4096] b44107e0 ~
  (4096, inf) 780b9833
* ``sm_103a`` ``fwd_h256``: (0, 8192] 0489c784 = (8192, 20480] bbb05314 =
  (20480, 40960] 1b284bae = (40960, 81920] 86d51eff = (81920, 114688] 466331ff =
  (114688, 196608] 86d51eff = (196608, 327680] ed9483d8 = (327680, 368640] 37f66318 =
  (368640, inf) b8f32dd4
* ``sm_103a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 3072] d447b7f6 = (3072, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_103a`` ``fwd_h512``: (0, 1024] 8757f6da ~ (1024, 4096] 7673978d =
  (4096, 16384] 3d20ab4a = (16384, inf) 3430e5d9
* ``sm_103a`` ``fwd_h5120``: (0, 640] 5eef28cd = (640, 6144] b5e173f5 =
  (6144, inf) 5eef28cd
* ``sm_103a`` ``fwd_w32_h5120``: (0, 640] 1defffdf = (640, 6144] 9ecb62c2 =
  (6144, inf) 9ecb62c2
* ``sm_103a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, inf) 288d7b1d
* ``sm_103a`` ``fwd_w32_h6144``: (0, 1024] 2763cd5e = (1024, 6144] 2763cd5e =
  (6144, inf) 5f89e5fd
* ``sm_103a`` ``fwd_residual_h6144``: (0, 256] 308da182 = (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
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

Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window ``above <
T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows; the table
below lists them).  The forward programs are the deterministic package's, with its
windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``, and at a
forward edge between programs of different per-row thread groups ``rstd`` moves by at
most two FP32 ulps (three on R200 at the residual H = 6144 edge 1024) and ``y`` by at
most one BF16 ulp, on up to about 32 % of the rows (measured on B200, GB300 and R200).
A backward window runs one of three forms.  Up to T 16 every backward route, and up to
64 rows ``bwd_h256`` and ``bwd_w32_h256``, run the one-cluster kernel, marked ``(one
cluster)``: one launch of 1 to 8 CTAs forming one thread-block cluster over every row,
with no workspace and no atomic, one program up to T 4 and one above (up to 4, 24 and 64
rows on the H = 256 routes).  Lower one-cluster tops, where a larger sampled token count
ran the backward, the forward and backward or the training step slower than the program
above: on sm_100a up to T 14 on ``bwd_w32_h2048`` and ``bwd_w32_h4096``, up to T 12 on
``bwd_h2048`` and ``bwd_h4096``, up to T 8 on ``bwd_h1024`` and ``bwd_w32_h1024``, up
to T 4 on ``bwd_h5120``, ``bwd_h6144``, ``bwd_w32_h5120``, ``bwd_w32_h6144`` and
``bwd_w32_residual_h6144`` and up to T 2 on ``bwd_h1536``, ``bwd_residual_h6144`` and
``bwd_w32_h1536``; on sm_103a up to T 12 on ``bwd_w32_h6144``, up to T 10 on
``bwd_w32_residual_h6144``, up to T 8 on ``bwd_w32_h1024``, up to T 6 on ``bwd_h6144``,
``bwd_residual_h6144`` and ``bwd_w32_h1536``, up to T 5 on ``bwd_h1536`` and
``bwd_w32_h5120`` and up to T 4 on ``bwd_h1024`` and ``bwd_h5120``; on sm_107a up to
T 15 on ``bwd_residual_h6144`` and ``bwd_w32_h5120``, up to T 12 on ``bwd_h5120``, up to
T 8 on ``bwd_h1024`` and up to T 4 on ``bwd_w32_h1280``.  No one-cluster window runs on
``bwd_h128``, ``bwd_w32_h128`` and ``bwd_h1280``, nor on sm_100a and sm_103a on
``bwd_w32_h1280``.  It forms
``dx`` with the single kernel's formula (its row sums in another order, so ``dx`` may
differ from the single kernel's by rounding) and ``dw`` in FP32 in a fixed order: the
same bits on every launch of a token count.  Where the notes below name token counts at
or below those tops, they describe the windows above them.  The single kernel computes
``dx`` row by row (bitwise reproducible) and adds the
``dw`` sums of its CTAs into FP32 column sums (FP32 adds) in a scheduling-dependent
order: its ``dw`` is not bitwise reproducible, even for a fixed ``T``.  The top window
of ``bwd_h256`` on sm_100a and sm_107a, on sm_103a of ``bwd_h2048``, ``bwd_h4096``,
``bwd_h5120``, ``bwd_h6144`` and ``bwd_residual_h6144`` and on sm_107a of
``bwd_h1536``, ``bwd_h2048``, ``bwd_h6144`` and ``bwd_residual_h6144`` (and of their
FP32-weight routes), the windows above T 49152 of sm_107a ``bwd_h512`` and above T 65536
of sm_107a ``bwd_h4096``, on sm_100a the windows above T 131072 of ``bwd_h2048``,
``bwd_h4096``, ``bwd_h6144`` and ``bwd_residual_h6144`` and above T 214016 of
``bwd_h1536`` (both weight dtypes), the lowest window of each of those routes that
runs its top window's program, the lowest window on sm_103a of ``bwd_h4096`` and
``bwd_w32_h4096``,
the window of ``bwd_w32_h512`` from T 227 to 292 on sm_100a, the sm_100a ``bwd_h256``
window from 20480 to 92160 rows, on sm_100a the windows of ``bwd_h4096`` from T 4608 to
16384 and from T 16384 to 36864 and of ``bwd_residual_h6144`` from T 1408 to 65536, run
a single kernel whose threads sum their ``dw`` columns in FP32 over every row they take
and add each sum to the FP32 sums once; its ``dw`` is not bitwise reproducible either.
In those sm_100a windows it is a faster variant of the single kernel of the route's top
window with the same ``dw`` arithmetic: on ``bwd_h4096`` with ``evict_last`` loads
(without read-ahead up to T 16384), on ``bwd_residual_h6144`` with 3 pipeline stages.
On sm_103a the window below the top window of ``bwd_w32_h4096``, ``bwd_h5120`` and
``bwd_w32_h5120``, on sm_107a the top window of ``bwd_h5120`` and ``bwd_w32_h5120``, on
sm_103a also the window from T 46592 to 53184 of ``bwd_h1536``, from T 68160 to 131072
of ``bwd_h2048`` and ``bwd_w32_h2048``, from T 83392 to 131072 of ``bwd_w32_h6144``,
and on sm_100a the windows from T 78784 to 131072 of ``bwd_h1536``, from T 77824 to
131072 of ``bwd_w32_h1536``, from T 65536 to 131072 of ``bwd_h2048``,
``bwd_w32_h2048``, ``bwd_h4096``, ``bwd_w32_h4096``, ``bwd_h6144``, ``bwd_w32_h6144``,
``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` and from T 56256 to 131072 of
``bwd_h5120`` and ``bwd_w32_h5120`` run a flush single kernel: each thread adds the
``dw`` products of a block of 8 rows in FP32 and adds each block sum to its FP32 column
sums (on sm_103a its windows end at T 262144 on ``bwd_w32_h4096``, T 196608 on
``bwd_w32_h5120`` and T 393216 on ``bwd_h5120``); its ``dw`` is not bitwise reproducible
either.  On sm_103a ``bwd_residual_h6144`` from T 1920 to 131072,
``bwd_w32_residual_h6144`` from T 1824 to 131072, ``bwd_h6144`` from T 24575 to 131072
and ``bwd_h4096`` from T 20480 to 262144, and on sm_107a ``bwd_h6144`` from T 30720 to
131072, ``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` from T 7168 to 131072,
``bwd_h4096`` from T 24576 to 65536 and ``bwd_w32_h5120`` from T 32767 to 131072 run a
part-row single kernel: each thread sums its ``dw`` columns in FP32 (on sm_107a
``bwd_w32_h5120`` in 8-row blocks), each CTA writes its sums as one FP32 row of a
scratch area of the workspace, and once every row is in, its helper CTAs (128; 96 on
sm_103a ``bwd_h6144`` up to T 65536, 256 on sm_107a ``bwd_h4096``) add the rows in FP32
and store ``dw``, with no atomic add of a sum; its ``dw`` is not bitwise reproducible
either (the row order follows scheduling).  On sm_103a from T 20480 to 65536 on
``bwd_h4096``, from T 24575 to 65536 on ``bwd_h6144`` and in every sm_107a window its
build cuts the chunk table for more slots per SM than the grid has CTAs (8 and 6 per SM
on sm_103a; 4 per SM on sm_107a, 8 on ``bwd_h4096``): more, shorter chunks, the grid
unchanged.  A window marked ``(det pair)`` runs the deterministic package's main kernel
and fixed-order dw reduce, rendered into this package; on ``bwd_h5120`` (both weight
dtypes) the det pair windows above T 36864 on sm_100a, 36301 on sm_103a and 19968 on
sm_107a run the deterministic package's programs of the rows above T 36864 (sm_100a),
36544 (sm_103a) and 19968 (sm_107a) up to T 56256 on sm_100a, up to T 61952 on sm_103a
and up to T 49152 on sm_107a (T 32767 on ``bwd_w32_h5120``), a single kernel above (on
sm_100a up to T 131072, the one-row-per-stage det pair above).  On sm_100a and sm_103a
``bwd_h1536``, ``bwd_w32_h1536``, ``bwd_h5120`` and ``bwd_w32_h5120`` run the
deterministic package's one-row-per-stage det pairs (each CTA one slot, rows strided
over the slots, eight SMs left out) from T 1184 to 77824 and from T 131072 to 214016
(H = 1536) and from T 767 to 36864 and above T 131072 (H = 5120) on sm_100a, and from
T 4096 to 46592 and above T 53184 (H = 1536) and from T 1024 to 36301 (H = 5120) on
sm_103a; below those sm_103a windows, from T 1216 to 4096 (H = 1536) and from T 192 to
1024 (H = 5120), they run on every SM: in records on B200 and GB200 (sm_100a) and on
GB300 (sm_103a) no sampled T in those windows had a backward call, a forward and
backward call or a checkpointed step slower than with the programs they replace.
The deterministic package's one-row-per-stage det pairs also run with that package's
slot plan (eight SMs left out) on sm_103a ``bwd_h2048`` and ``bwd_w32_h2048`` from
T 171 to 16384, ``bwd_h6144`` from T 896 to 4096 and ``bwd_h4096`` and
``bwd_w32_h4096`` from T 1024 to 20480.  On ``bwd_h128`` and ``bwd_w32_h128`` the
deterministic package's ``rpw`` det pair (each CTA one slot, 16-row tiles strided over
the slots) runs from T 2048 to 8192 on sm_100a and sm_103a (eight SMs left out) and
from T 2047 to 65536 on sm_107a (every SM).  On ``bwd_h1024`` the deterministic
package's one-row-per-stage det pairs (every SM) also run above T 32768 on sm_100a and
sm_103a, and on ``bwd_w32_h1024`` above T 32768 on sm_100a.  On sm_107a the
deterministic package's grouped det pairs (five-row tiles strided over the slots, one
SM left out) run on ``bwd_h1024`` and ``bwd_w32_h1024`` above T 4095 and on
``bwd_h1280`` and ``bwd_w32_h1280`` above T 6143 (three-row tiles above T 32767).
On sm_103a ``bwd_h256`` (both weight dtypes) every window above 64
rows runs a det pair.  Their main kernels are those the deterministic package runs for
those rows on a device with 152 or more SMs, except from 1080 to 2048 rows, where it is
the main kernel that package runs from 2048 to 5120 rows on sm_100a; their fixed-order
dw reduce is that package's, except from 720896 to 1572864 rows, where it is another of
that package's dw reduces.  On sm_103a ``bwd_h256`` and ``bwd_w32_h256`` run the
deterministic package's one-row-per-stage det pair (each CTA one slot, eight-row tiles
strided over the slots, eight SMs left out) from 2048 to 262144 rows.  On ``bwd_h256``
(both weight dtypes) the det pair the deterministic package runs above 262144 rows on
sm_100a also runs from 92160 to 524288 rows on sm_100a, and with FP32 weight up to
864240 rows.  The single kernel of the top window runs above.  On sm_103a
``bwd_w32_h512`` runs from T
1024 to 32768 the det pair the deterministic package runs from T 4096.  On sm_100a
``bwd_h512`` above T 4096, ``bwd_h2048`` from T 13608 to 58880, ``bwd_w32_h2048`` from T
58880 to 65536 (the FP32-weight det pair) and ``bwd_h6144`` from T 5632 to 42496 run the
det pair the deterministic package runs there, and ``bwd_h2048`` from T 58880 to 65536
the flush single kernel of its window above.  On sm_100a ``bwd_h1536`` from T 77824 to
78784 runs the det pair the deterministic package runs above T 161792, ``bwd_h4096``
from T 44032 to 65536 the det pair that package runs there, and ``bwd_w32_h4096`` from
T 44032 to 65536 the FP32-weight det pair that package runs there.  On sm_103a
``bwd_w32_h4096``
from T 20480 to 65536 runs FP32-weight det pairs of the deterministic package, split at T
48640 (above it the twin of that package's 512-thread row-issuer program).  On sm_100a
``bwd_w32_h5120`` from T 36864 to 56256 runs the FP32-pair det pair, which that package
runs at every T above 36864; above T 56256 the route runs the flush single kernel (to T
131072), which sums in plain FP32 words, and the one-row-per-stage det pair above.  On
sm_107a ``bwd_h4096`` from T 24064 to 24576 and ``bwd_w32_h5120``
from T 4096
to 32767 run the det pair the deterministic package runs there, and ``bwd_w32_h4096``
from T 24064 to 65536 the FP32-weight det pair the deterministic package's
``bwd_w32_h4096`` runs from T 24064 (that package cuts it at T 50176).  On sm_107a
``bwd_h1536`` and ``bwd_w32_h1536`` from T 45056, the deterministic package's top edge,
to 65536 run the
det pair that package runs there (its four-row block form).  On sm_107a ``bwd_h2048``
from T 50688 to 65536 runs the det pair the deterministic package runs there (its
eight-row block form), for speed.  On sm_100a ``bwd_w32_h1536`` from T 1024 to 1184,
``bwd_w32_h2048`` from T 1024 to 4096 and ``bwd_w32_h4096`` from T 1792 to 3584 run the
single kernel of their top window and ``bwd_w32_h2048`` from T 336 to 1024 a column-task
kernel (its CTAs first sum whole columns of ``dw`` over every row); on sm_107a
``bwd_w32_h6144`` from T 19456 to 28672 runs the det pair the deterministic package runs
above T 19456, ``bwd_w32_h2048`` from T 1024 to 16384 the single kernel of its largest
token counts, and ``bwd_w32_h1536`` from T 1024 to 16384 and ``bwd_w32_h6144`` from T
2304 to 19456 the single kernel of their top
window.  On sm_100a ``bwd_w32_h2048`` from T 64 to 336 and on sm_107a ``bwd_w32_h5120``
from T 896 to 1280 run the deterministic package's column-task program of those token
counts, and on sm_103a ``bwd_w32_h512`` from T 960 to 1024 the FP32-weight det pair the
deterministic package runs there: in a same-node comparison on B200, GB300 and VR200
(CUDA-graph span, L2 hot and flushed before every call) each took 0.54x to 0.97x the
time of the program it replaces at every sampled token
count.  On sm_100a ``bwd_h2048`` from T 45056 to 58880
runs the single kernel of its window from T 7168 to 11264 and ``bwd_w32_h2048`` from T
45056 to 58880 the single kernel of its neighbouring windows, within 2 % of the det
pair's time.  On sm_107a ``bwd_h6144`` from T 19456 to 30720 runs the det pair the
deterministic package runs there.  On sm_100a ``bwd_h512`` and ``bwd_w32_h512`` up to T
227 run the deterministic package's column-task program of that window, which is faster
there.  On sm_103a ``bwd_h512`` up to T 64, ``bwd_w32_h512`` up to T 227, ``bwd_h1536``
and ``bwd_w32_h1536`` from T 64 to 214, ``bwd_h2048`` from T 63 and ``bwd_w32_h2048``
from T 64 to 171, ``bwd_h4096`` and ``bwd_w32_h4096`` from T 32 to 152 and T 32 of
``bwd_residual_h6144`` and ``bwd_w32_residual_h6144`` run the deterministic package's
column-task programs of those token counts, and ``bwd_w32_h512`` from T 227 to 292 and
``bwd_w32_h1536`` from T 214 to 320 the det pair of the window above, which are faster
there.  On sm_103a ``bwd_residual_h6144`` from T 56 to 304 and
``bwd_w32_residual_h6144`` from T 48 to 320 run the deterministic package's column-task
program of those token counts (and its FP32-weight twin), which that package runs from T
0 to 1216: on GB300 it took 5.82, 8.13 and 9.34 us at T 64, 256 and 304 where the
whole-SM det pair took 7.65, 10.24 and 10.53 us.  On sm_103a ``bwd_h5120`` from T 17 to
64, ``bwd_w32_h5120`` from T 17 to 192, ``bwd_h6144`` from T 17 to 896 and
``bwd_w32_h6144`` from T 17 to 1024 run the deterministic package's column-task program
of the route in its ticket form (the form whose ``dx`` and ``dw`` the fixed-claim form
reproduces bit for bit): on GB300 (benchmark, CUDA-graph span, L2 flushed before every
call; one or two runs per T) no sampled T up to those token counts had a backward call,
a forward and backward call or a checkpointed step (two forward calls and a backward
call) slower than with the programs it replaces beyond the run-to-run band in all its
runs.  Above them the det pair of the window above runs: at T 96 on ``bwd_h5120``, T 256
on ``bwd_w32_h5120`` and T 1024 on ``bwd_h6144`` (and at some larger sampled T) a
forward and backward call or a checkpointed step with the ticket form was slower in both
runs, though its backward call alone was faster there.  On sm_107a ``bwd_h512`` up to T
576, ``bwd_w32_h512`` up to T 384, ``bwd_h1536`` from T 15 to 320, ``bwd_w32_h1536``
from T 1 to 320, ``bwd_h2048`` from T 31 and ``bwd_w32_h2048`` up to T 256,
``bwd_h4096`` from T 1 and ``bwd_w32_h4096`` up to T 256, ``bwd_h5120`` from T 31 to
256, from T 511 to 640 and T 896, ``bwd_w32_h5120`` from T 15 to 256, from T 511 to 640
and T 896, ``bwd_h6144`` from T 15 to 960 and T 1152, ``bwd_w32_h6144`` from T 1 to
1664, ``bwd_residual_h6144`` from T 15 to 320 and from T 447 to 576 and
``bwd_w32_residual_h6144`` up to T 768 run the deterministic package's column-task
programs of those token counts, which took 0.53x to 0.97x the time of the programs they
replace at every sampled T in two R200 records with 212 SMs (the slower record per T)
and were also faster there in a VR200 screen.  On sm_107a ``bwd_h256`` and
``bwd_w32_h256`` up to 786432 rows run the deterministic package's det pair of those
rows, which took 0.76x to 0.95x the time of the program it replaces at every sampled row
count in two R200 records with 212 SMs and was also faster in a VR200 screen; above
786432 rows the single kernel of the top window runs.  On 9 backward routes the lowest
window, from T 1 up to a token count from 1 to 31, runs a single kernel.  On
``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` (and their FP32-weight routes) the window above the one-cluster
windows, from T 17 up to T 64 (on sm_103a up to T 63 on ``bwd_h2048``, up to T 32 on
``bwd_h4096`` and ``bwd_w32_h4096``, up to T 48 on ``bwd_residual_h6144`` and
``bwd_w32_residual_h6144``, except T 32 of those two, and not on ``bwd_h5120``,
``bwd_w32_h5120``, ``bwd_h6144`` and ``bwd_w32_h6144``; on sm_107a only on ``bwd_h2048``
and ``bwd_h5120``, up to T 31), runs the fixed-claim form of the column-task kernel:
each CTA takes its rows and its column tasks by its index instead of by claim tickets,
so no CTA waits for another, and its ``dx`` and ``dw`` are bitwise those of the ticket
form.  Every program sums in plain FP32 words, except the FP32-pair det pairs above,
which add each block of rows into an FP32 pair (a sum and its compensation word).  The
tests hold no tolerance: they judge every floating output by its error against an FP64
reference beside the error of the naive PyTorch implementation (the same ops in BF16 on
the same BF16 inputs), over 32 independent random inputs of each case: in the L2 norm
and in the max norm the median and the largest error at most 1.005 times the naive's,
and 0 where the naive's is 0.

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge: ``=`` bitwise-identical ``dx`` on the common rows across it (``dw``
sums every row and is not bitwise reproducible), ``~`` the forward's ulp differences
(see above), ``?`` not yet measured.

The FP32-weight routes (``bwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows and are not listed; a route re-cut for the FP32
weight (another twin in a window) is listed with its own windows.

A backward window marked ``(det pair)`` runs the deterministic package's main kernel
(the program named) and its dw reduce, one marked ``(one cluster)`` the one-cluster
kernel of small token counts; the others run the single kernel.

Beside a kernel that holds shared memory on every SM (one CTA per SM with a large
dynamic shared-memory allocation), a backward window waits until that kernel exits
when the shared memory it leaves is too small for the window's program.  Beside
those, leave some SMs free of the resident kernel.  Measured with
100 / 132 KB per CTA (``SMEM_WAITS``): ``>= K`` after a window means it waits beside
K KB or more, ``-`` that it starts beside every measured size; an architecture or
route not listed is unmeasured.

* ``sm_100a`` ``bwd_h1024``: (0, 4] -, (4, 8] -, (8, 64] -, (64, 128] -, (128, 1024] -, (1024, 2048] -, (2048, 32768] -, (32768, inf) -
* ``sm_100a`` ``bwd_h128``: (0, 1024] -, (1024, 2048] -, (2048, 8192] -, (8192, 12288] -, (12288, 16384] -, (16384, inf) -
* ``sm_100a`` ``bwd_h1280``: (0, 1] -, (1, 128] -, (128, 512] -, (512, 4096] -, (4096, 8192] -, (8192, inf) -
* ``sm_100a`` ``bwd_h1536``: (0, 2] >= 100, (2, 64] -, (64, 384] -, (384, 1024] >= 132, (1024, 1184] >= 100, (1184, 77824] >= 132, (77824, 78784] >= 100, (78784, 131072] -, (131072, 214016] >= 132, (214016, inf) -
* ``sm_100a`` ``bwd_h2048``: (0, 4] >= 100, (4, 12] >= 100, (12, 64] -, (64, 336] -, (336, 3072] >= 100, (3072, 4096] >= 100, (4096, 7168] >= 100, (7168, 11264] >= 132, (11264, 16384] >= 100, (16384, 58880] >= 100, (58880, 131072] -, (131072, inf) >= 132
* ``sm_100a`` ``bwd_h256``: (0, 4] >= 100, (4, 24] >= 100, (24, 64] >= 100, (64, 20480] -, (20480, 92160] -, (92160, 524288] -, (524288, inf) -
* ``sm_100a`` ``bwd_h4096``: (0, 4] >= 100, (4, 12] >= 100, (12, 64] -, (64, 1152] -, (1152, 4096] >= 132, (4096, 4608] >= 132, (4608, 16384] >= 100, (16384, 36864] >= 100, (36864, 44032] >= 132, (44032, 65536] >= 100, (65536, 131072] -, (131072, inf) -
* ``sm_100a`` ``bwd_h512``: (0, 4] >= 100, (4, 16] >= 100, (16, 227] -, (227, 1024] -, (1024, 1280] >= 132, (1280, 4096] -, (4096, 32768] >= 100, (32768, inf) >= 100
* ``sm_100a`` ``bwd_h5120``: (0, 4] >= 100, (4, 64] >= 100, (64, 767] -, (767, 36864] -, (36864, 56256] -, (56256, 131072] >= 132, (131072, inf) -
* ``sm_100a`` ``bwd_h6144``: (0, 4] >= 100, (4, 64] -, (64, 1152] >= 132, (1152, 4096] >= 132, (4096, 42496] >= 100, (42496, 65536] >= 100, (65536, 131072] >= 100, (131072, inf) >= 100
* ``sm_100a`` ``bwd_residual_h6144``: (0, 2] >= 100, (2, 64] -, (64, 1408] >= 100, (1408, 65536] >= 100, (65536, 131072] >= 100, (131072, inf) >= 100
* ``sm_100a`` ``bwd_w32_h1024``: (0, 4] -, (4, 8] -, (8, 1024] -, (1024, 2048] -, (2048, 4096] -, (4096, 12288] -, (12288, 32768] -, (32768, inf) -
* ``sm_100a`` ``bwd_w32_h128``: (0, 1024] -, (1024, 1536] -, (1536, 2048] -, (2048, 8192] -, (8192, 12288] -, (12288, 16384] -, (16384, inf) -
* ``sm_100a`` ``bwd_w32_h1280``: (0, 1] -, (1, 128] -, (128, 512] -, (512, 4096] -, (4096, inf) -
* ``sm_100a`` ``bwd_w32_h1536``: (0, 2] >= 100, (2, 64] -, (64, 384] -, (384, 1024] >= 132, (1024, 1184] -, (1184, 77824] >= 132, (77824, 131072] -, (131072, 214016] >= 132, (214016, inf) -
* ``sm_100a`` ``bwd_w32_h2048``: (0, 4] >= 100, (4, 14] >= 100, (14, 64] -, (64, 336] -, (336, 1024] -, (1024, 4096] >= 132, (4096, 15403] >= 100, (15403, 45056] >= 132, (45056, 58880] >= 132, (58880, 65536] >= 100, (65536, 131072] -, (131072, inf) >= 132
* ``sm_100a`` ``bwd_w32_h256``: (0, 4] >= 100, (4, 24] >= 100, (24, 64] >= 100, (64, 20480] -, (20480, 92160] -, (92160, 864240] -, (864240, inf) -
* ``sm_100a`` ``bwd_w32_h4096``: (0, 4] >= 100, (4, 14] >= 100, (14, 64] -, (64, 1792] -, (1792, 3584] -, (3584, 36864] -, (36864, 44032] >= 132, (44032, 65536] >= 100, (65536, 131072] -, (131072, inf) -
* ``sm_100a`` ``bwd_w32_h512``: (0, 4] >= 100, (4, 16] >= 100, (16, 227] -, (227, 292] -, (292, 960] -, (960, inf) -
* ``sm_100a`` ``bwd_w32_h5120``: (0, 4] >= 100, (4, 64] >= 100, (64, 767] -, (767, 36864] -, (36864, 56256] >= 100, (56256, 131072] >= 132, (131072, inf) -
* ``sm_100a`` ``bwd_w32_h6144``: (0, 4] >= 100, (4, 64] -, (64, 1168] >= 132, (1168, 3584] >= 132, (3584, 65536] >= 100, (65536, 131072] >= 100, (131072, inf) >= 100
* ``sm_100a`` ``bwd_w32_residual_h6144``: (0, 4] >= 100, (4, 64] -, (64, 1408] >= 100, (1408, 65536] >= 100, (65536, 131072] >= 100, (131072, inf) >= 100
* ``sm_103a`` ``bwd_h1024``: (0, 4] -, (4, 64] -, (64, 128] -, (128, 1024] -, (1024, 1536] -, (1536, 2048] -, (2048, 32768] -, (32768, inf) -
* ``sm_103a`` ``bwd_h128``: (0, 1024] -, (1024, 2048] -, (2048, 8192] -, (8192, 12288] -, (12288, 16384] -, (16384, inf) -
* ``sm_103a`` ``bwd_h1280``: (0, 1] -, (1, 128] -, (128, 512] -, (512, 2048] -, (2048, inf) -
* ``sm_103a`` ``bwd_h1536``: (0, 4] >= 100, (4, 5] >= 100, (5, 64] -, (64, 214] -, (214, 1024] >= 132, (1024, 1216] >= 100, (1216, 4096] >= 132, (4096, 46592] >= 132, (46592, 53184] -, (53184, inf) >= 132
* ``sm_103a`` ``bwd_h2048``: (0, 4] >= 100, (4, 16] >= 100, (16, 63] -, (63, 171] -, (171, 16384] >= 132, (16384, 32768] >= 100, (32768, 58496] >= 100, (58496, 61952] >= 100, (61952, 68160] >= 100, (68160, 131072] -, (131072, inf) >= 132
* ``sm_103a`` ``bwd_h256``: (0, 4] >= 100, (4, 24] >= 100, (24, 64] >= 100, (64, 1080] -, (1080, 2048] -, (2048, 262144] -, (262144, 720896] -, (720896, 1572864] -, (1572864, inf) -
* ``sm_103a`` ``bwd_h4096``: (0, 4] >= 100, (4, 16] >= 100, (16, 32] -, (32, 152] -, (152, 256] -, (256, 1024] -, (1024, 20480] -, (20480, 65536] >= 100, (65536, 262144] >= 100, (262144, inf) -
* ``sm_103a`` ``bwd_h512``: (0, 4] >= 100, (4, 16] >= 100, (16, 64] -, (64, 1024] -, (1024, 4096] -, (4096, 32768] -, (32768, inf) -
* ``sm_103a`` ``bwd_h5120``: (0, 4] >= 100, (4, 16] >= 100, (16, 64] >= 100, (64, 192] >= 132, (192, 1024] -, (1024, 36301] -, (36301, 61952] >= 132, (61952, 393216] >= 132, (393216, inf) >= 132
* ``sm_103a`` ``bwd_h6144``: (0, 4] >= 100, (4, 6] >= 100, (6, 16] -, (16, 896] >= 100, (896, 4096] >= 132, (4096, 24575] >= 132, (24575, 65536] >= 100, (65536, 131072] >= 100, (131072, inf) >= 100
* ``sm_103a`` ``bwd_residual_h6144``: (0, 4] >= 100, (4, 6] >= 100, (6, 31] -, (31, 32] >= 100, (32, 48] -, (48, 56] >= 100, (56, 304] >= 100, (304, 1920] >= 100, (1920, 131072] >= 100, (131072, inf) >= 100
* ``sm_103a`` ``bwd_w32_h1024``: (0, 4] -, (4, 8] -, (8, 1024] -, (1024, 1536] -, (1536, 2048] -, (2048, 4096] -, (4096, 12288] -, (12288, 32768] -, (32768, 131072] -, (131072, inf) -
* ``sm_103a`` ``bwd_w32_h128``: (0, 1024] -, (1024, 2048] -, (2048, 8192] -, (8192, 12288] -, (12288, 16384] -, (16384, inf) -
* ``sm_103a`` ``bwd_w32_h1280``: (0, 1] -, (1, 128] -, (128, 512] -, (512, 4096] -, (4096, inf) -
* ``sm_103a`` ``bwd_w32_h1536``: (0, 4] >= 100, (4, 6] >= 100, (6, 64] -, (64, 214] -, (214, 1024] >= 132, (1024, 1216] >= 100, (1216, 4096] >= 132, (4096, 46592] >= 132, (46592, 53184] >= 100, (53184, inf) >= 132
* ``sm_103a`` ``bwd_w32_h2048``: (0, 4] >= 100, (4, 16] >= 100, (16, 64] -, (64, 171] -, (171, 16384] >= 132, (16384, 32768] >= 100, (32768, 58496] >= 100, (58496, 61952] >= 100, (61952, 68160] >= 100, (68160, 131072] -, (131072, inf) >= 132
* ``sm_103a`` ``bwd_w32_h256``: (0, 4] >= 100, (4, 24] >= 100, (24, 64] >= 100, (64, 1080] -, (1080, 2048] -, (2048, 262144] -, (262144, 720896] -, (720896, 1572864] -, (1572864, inf) -
* ``sm_103a`` ``bwd_w32_h4096``: (0, 4] >= 100, (4, 16] >= 100, (16, 32] -, (32, 152] -, (152, 256] -, (256, 1024] -, (1024, 20480] -, (20480, 48640] >= 132, (48640, 65536] >= 100, (65536, 262144] -, (262144, inf) -
* ``sm_103a`` ``bwd_w32_h512``: (0, 4] >= 100, (4, 16] >= 100, (16, 227] -, (227, 1024] -, (1024, 32768] -, (32768, inf) -
* ``sm_103a`` ``bwd_w32_h5120``: (0, 4] >= 100, (4, 5] >= 100, (5, 16] >= 100, (16, 192] >= 100, (192, 1024] -, (1024, 36301] -, (36301, 61952] >= 132, (61952, 196608] >= 132, (196608, inf) >= 132
* ``sm_103a`` ``bwd_w32_h6144``: (0, 4] >= 100, (4, 12] >= 100, (12, 16] -, (16, 1024] >= 100, (1024, 4096] >= 132, (4096, 49152] >= 132, (49152, 83392] >= 132, (83392, 131072] >= 100, (131072, inf) >= 100
* ``sm_103a`` ``bwd_w32_residual_h6144``: (0, 4] >= 100, (4, 10] >= 100, (10, 31] -, (31, 32] >= 100, (32, 48] -, (48, 320] >= 100, (320, 1824] >= 100, (1824, 131072] >= 100, (131072, inf) >= 100
* ``sm_107a`` ``bwd_h1024``: (0, 4] -, (4, 8] -, (8, 1536] -, (1536, 2048] -, (2048, 4095] -, (4095, inf) -
* ``sm_107a`` ``bwd_h128``: (0, 1024] -, (1024, 2047] -, (2047, 65536] -, (65536, inf) -
* ``sm_107a`` ``bwd_h1280``: (0, 1] -, (1, 128] -, (128, 512] -, (512, 6143] -, (6143, 32767] -, (32767, inf) -
* ``sm_107a`` ``bwd_h1536``: (0, 4] >= 100, (4, 16] >= 100, (16, 320] -, (320, 1024] -, (1024, 12288] >= 100, (12288, 16384] >= 100, (16384, 45056] >= 100, (45056, 65536] >= 100, (65536, inf) -
* ``sm_107a`` ``bwd_h2048``: (0, 4] >= 100, (4, 16] >= 100, (16, 31] -, (31, 256] -, (256, 1024] >= 100, (1024, 16384] >= 100, (16384, 50176] >= 100, (50176, 50688] -, (50688, 65536] >= 100, (65536, inf) -
* ``sm_107a`` ``bwd_h256``: (0, 4] >= 100, (4, 24] >= 100, (24, 64] >= 100, (64, 786432] -, (786432, inf) -
* ``sm_107a`` ``bwd_h4096``: (0, 4] >= 100, (4, 16] >= 100, (16, 256] -, (256, 1024] >= 100, (1024, 4096] >= 100, (4096, 24064] >= 100, (24064, 24576] >= 100, (24576, 65536] >= 100, (65536, inf) >= 132
* ``sm_107a`` ``bwd_h512``: (0, 4] >= 100, (4, 16] >= 100, (16, 576] -, (576, 1024] -, (1024, 4096] >= 100, (4096, 16384] >= 100, (16384, 49152] >= 100, (49152, inf) -
* ``sm_107a`` ``bwd_h5120``: (0, 4] >= 100, (4, 12] >= 100, (12, 31] >= 100, (31, 256] -, (256, 511] >= 100, (511, 640] -, (640, 895] >= 100, (895, 896] -, (896, 4096] >= 100, (4096, 19968] >= 100, (19968, 49152] >= 100, (49152, inf) >= 132
* ``sm_107a`` ``bwd_h6144``: (0, 4] >= 100, (4, 16] >= 100, (16, 960] >= 132, (960, 1151] >= 100, (1151, 1152] >= 132, (1152, 19456] >= 100, (19456, 30720] >= 100, (30720, 131072] >= 100, (131072, inf) >= 100
* ``sm_107a`` ``bwd_residual_h6144``: (0, 4] >= 100, (4, 15] >= 100, (15, 320] >= 100, (320, 447] >= 100, (447, 576] >= 100, (576, 3584] >= 100, (3584, 7168] >= 100, (7168, 131072] >= 100, (131072, inf) >= 100
* ``sm_107a`` ``bwd_w32_h1024``: (0, 4] -, (4, 16] -, (16, 1536] -, (1536, 4095] -, (4095, inf) -
* ``sm_107a`` ``bwd_w32_h128``: (0, 1024] -, (1024, 2047] -, (2047, 65536] -, (65536, inf) -
* ``sm_107a`` ``bwd_w32_h1280``: (0, 4] -, (4, 128] -, (128, 512] -, (512, 4096] -, (4096, 6143] -, (6143, 32767] -, (32767, inf) -
* ``sm_107a`` ``bwd_w32_h1536``: (0, 4] >= 100, (4, 16] >= 100, (16, 320] -, (320, 1024] -, (1024, 16384] -, (16384, 45056] >= 100, (45056, 65536] >= 100, (65536, inf) -
* ``sm_107a`` ``bwd_w32_h2048``: (0, 4] >= 100, (4, 16] >= 100, (16, 256] -, (256, 1024] >= 100, (1024, 16384] -, (16384, 50176] >= 100, (50176, inf) -
* ``sm_107a`` ``bwd_w32_h256``: (0, 4] >= 100, (4, 24] >= 100, (24, 64] >= 100, (64, 786432] -, (786432, inf) -
* ``sm_107a`` ``bwd_w32_h4096``: (0, 4] >= 100, (4, 16] >= 100, (16, 256] -, (256, 1024] >= 100, (1024, 4096] >= 100, (4096, 24064] >= 100, (24064, 65536] >= 100, (65536, inf) >= 132
* ``sm_107a`` ``bwd_w32_h512``: (0, 4] >= 100, (4, 16] >= 100, (16, 384] -, (384, 1024] -, (1024, 4096] -, (4096, 16384] >= 100, (16384, 49152] >= 100, (49152, inf) -
* ``sm_107a`` ``bwd_w32_h5120``: (0, 4] >= 100, (4, 15] >= 100, (15, 256] -, (256, 511] >= 100, (511, 640] -, (640, 895] >= 100, (895, 1280] -, (1280, 4096] >= 100, (4096, 19968] >= 100, (19968, 32767] >= 100, (32767, 131072] >= 100, (131072, inf) >= 132
* ``sm_107a`` ``bwd_w32_h6144``: (0, 4] >= 100, (4, 16] >= 100, (16, 1664] >= 132, (1664, 2304] >= 132, (2304, 19456] >= 100, (19456, 28672] >= 100, (28672, inf) >= 100
* ``sm_107a`` ``bwd_w32_residual_h6144``: (0, 4] >= 100, (4, 16] >= 100, (16, 768] >= 100, (768, 3584] >= 100, (3584, 7168] >= 100, (7168, 131072] >= 100, (131072, inf) >= 100

Programs that need a whole SM wait beside any kernel resident on every SM, whatever
its carveout: the deterministic package's whole-SM programs where this package runs
them (``WHOLE_SM``: det-pair windows and forward programs; no single-kernel window).
Beside those, leave some SMs free of the resident kernel.

* ``sm_107a`` ``bwd_h1536``: (12288, 65536]
* ``sm_107a`` ``bwd_w32_h1536``: (16384, 65536]
* ``sm_107a`` ``bwd_h2048``: (256, 1024], (16384, 50176], (50688, 65536]
* ``sm_107a`` ``bwd_w32_h2048``: (256, 1024], (16384, 50176]
* ``sm_107a`` ``bwd_h4096``: (1024, 24064]
* ``sm_107a`` ``bwd_w32_h4096``: (1024, 65536]
* ``sm_107a`` ``bwd_residual_h6144``: (320, 447], (3584, 7168]
* ``sm_107a`` ``bwd_w32_residual_h6144``: (3584, 7168]

* ``sm_100a`` ``bwd_h128``: (0, 1024] 93466008 ? (1024, 2048] 401a018c ?
  (2048, 8192] d197bcc3 (det pair) ? (8192, 12288] 8b0b86c8 ? (12288, 16384] c50d3bbd ?
  (16384, inf) 8b0b86c8
* ``sm_100a`` ``bwd_w32_h128``: (0, 1024] ab8a320c ? (1024, 1536] b5f8c6c9 ?
  (1536, 2048] a45d09ae ? (2048, 8192] e23083c7 (det pair) ? (8192, 12288] 3e751b6f ?
  (12288, 16384] d0fe504e ? (16384, inf) 3e751b6f
* ``sm_100a`` ``bwd_h256``: (0, 4] 780c64c6 (one cluster) ?
  (4, 24] f2ccf719 (one cluster) ? (24, 64] 9fb6b410 (one cluster) ?
  (64, 20480] 195c4317 = (20480, 92160] 44ef3141 = (92160, 524288] a385ae16 (det pair) =
  (524288, inf) 44ef3141
* ``sm_100a`` ``bwd_w32_h256``: (0, 4] 67cb5641 (one cluster) ?
  (4, 24] aa745395 (one cluster) ? (24, 64] 6a0114cb (one cluster) ?
  (64, 20480] 8de01314 = (20480, 92160] b71bbe98 = (92160, 864240] 420788e6 (det pair) ?
  (864240, inf) b71bbe98
* ``sm_100a`` ``bwd_h512``: (0, 4] 64a29969 (one cluster) ?
  (4, 16] c167e0d2 (one cluster) ? (16, 227] 931757c8 = (227, 1024] 09601d42 (det pair)
  = (1024, 1280] 123db0b2 (det pair) = (1280, 4096] d0bde38a ?
  (4096, 32768] ad1a5a3c (det pair) ? (32768, inf) 8877322a (det pair)
* ``sm_100a`` ``bwd_w32_h512``: (0, 4] 8c261dfc (one cluster) ?
  (4, 16] 88e6252b (one cluster) ? (16, 227] 404c8f54 = (227, 292] 1a322bb1 =
  (292, 960] 9f07cea1 (det pair) = (960, inf) fd08c6f4
* ``sm_100a`` ``bwd_h1024``: (0, 4] 330930e9 (one cluster) ?
  (4, 8] 28647b7c (one cluster) ? (8, 64] 89fa3e96 ? (64, 128] 5ad5f096 ?
  (128, 1024] 89fa3e96 ? (1024, 2048] 0c581a13 (det pair) ? (2048, 32768] af798d10 ?
  (32768, inf) acd331db (det pair)
* ``sm_100a`` ``bwd_w32_h1024``: (0, 4] 81e742d5 (one cluster) ?
  (4, 8] 8501e95e (one cluster) ? (8, 1024] 744d7429 ? (1024, 2048] 561bea88 (det pair)
  ? (2048, 4096] 1a45ade5 ? (4096, 12288] 1a45ade5 ? (12288, 32768] 1a45ade5 ?
  (32768, inf) b201ba67 (det pair)
* ``sm_100a`` ``bwd_h1280``: (0, 1] 8895b022 ? (1, 128] a5f2942d ? (128, 512] 809235a3 ?
  (512, 4096] 12bbedff (det pair) ? (4096, 8192] c646e097 (det pair) ?
  (8192, inf) b2cf87e6
* ``sm_100a`` ``bwd_w32_h1280``: (0, 1] 103805d9 ? (1, 128] 8d8ef930 ?
  (128, 512] e476d0b2 ? (512, 4096] 80956d25 (det pair) ? (4096, inf) f6956a41
* ``sm_100a`` ``bwd_h1536``: (0, 2] db218c92 (one cluster) ? (2, 64] 02e761bc =
  (64, 384] dc478ac1 = (384, 1024] 7dc29629 (det pair) =
  (1024, 1184] c1a9efdf (det pair) ? (1184, 77824] 38e32d90 (det pair) ?
  (77824, 78784] b457c4d1 (det pair) ? (78784, 131072] 0d5dc239 ?
  (131072, 214016] 38e32d90 (det pair) ? (214016, inf) ae610afd
* ``sm_100a`` ``bwd_w32_h1536``: (0, 2] 6d3986ee (one cluster) ? (2, 64] 166e288c =
  (64, 384] e83ee45f = (384, 1024] baff9ecb (det pair) = (1024, 1184] 4b6981c1 ?
  (1184, 77824] e30c1899 (det pair) ? (77824, 131072] 363eabb3 ?
  (131072, 214016] e30c1899 (det pair) ? (214016, inf) 4b6981c1
* ``sm_100a`` ``bwd_h2048``: (0, 4] c5f62407 (one cluster) ?
  (4, 12] a951623c (one cluster) ? (12, 64] 518cf16a = (64, 336] 64e6aa82 =
  (336, 3072] caf6978e (det pair) = (3072, 4096] 9ac8d21e (det pair) =
  (4096, 7168] c73074cf (det pair) = (7168, 11264] 0fa4f255 =
  (11264, 16384] c73074cf (det pair) ? (16384, 58880] ceeef043 (det pair) ?
  (58880, 131072] 8404aa43 = (131072, inf) 0fa4f255
* ``sm_100a`` ``bwd_w32_h2048``: (0, 4] 0017ab41 (one cluster) ?
  (4, 14] 65e5f202 (one cluster) ? (14, 64] c2f08c95 = (64, 336] 9030dbeb =
  (336, 1024] 17493054 = (1024, 4096] ba7dbf6a = (4096, 15403] b187e598 (det pair) =
  (15403, 45056] ba7dbf6a = (45056, 58880] ba7dbf6a ? (58880, 65536] 7e6a0b63 (det pair)
  ? (65536, 131072] 54f6e4b8 = (131072, inf) ba7dbf6a
* ``sm_100a`` ``bwd_h4096``: (0, 4] 945a0afd (one cluster) ?
  (4, 12] 41f44b4b (one cluster) ? (12, 64] 82b84544 = (64, 1152] 4822793e =
  (1152, 4096] 9a830956 (det pair) = (4096, 4608] 35a0b0ca (det pair) ?
  (4608, 16384] cf4c213e ? (16384, 36864] 681b33d9 ? (36864, 44032] 35a0b0ca (det pair)
  ? (44032, 65536] 35a0b0ca (det pair) ? (65536, 131072] cda582bb =
  (131072, inf) 15e157d2
* ``sm_100a`` ``bwd_w32_h4096``: (0, 4] f6bd1634 (one cluster) ?
  (4, 14] b378ba00 (one cluster) ? (14, 64] cdef0b34 = (64, 1792] 86160705 =
  (1792, 3584] f7573bcb = (3584, 36864] f7573bcb = (36864, 44032] 626aed74 (det pair) ?
  (44032, 65536] 626aed74 (det pair) ? (65536, 131072] 3a10d379 = (131072, inf) f7573bcb
* ``sm_100a`` ``bwd_h5120``: (0, 4] dbc680f3 (one cluster) ? (4, 64] 29bcbde1 =
  (64, 767] 080e0a6c ? (767, 36864] bfde9b88 (det pair) ?
  (36864, 56256] b482c84d (det pair) = (56256, 131072] 670f4caf ?
  (131072, inf) bfde9b88 (det pair)
* ``sm_100a`` ``bwd_w32_h5120``: (0, 4] d4713822 (one cluster) ? (4, 64] cb653957 =
  (64, 767] 139d162a ? (767, 36864] f9b2d42c (det pair) ?
  (36864, 56256] 9479b129 (det pair) ? (56256, 131072] 13bd0c5f ?
  (131072, inf) f9b2d42c (det pair)
* ``sm_100a`` ``bwd_h6144``: (0, 4] e37752b0 (one cluster) ? (4, 64] e6475d8f =
  (64, 1152] 44508df7 = (1152, 4096] 5c804aed (det pair) =
  (4096, 42496] b93bf060 (det pair) ? (42496, 65536] 9433061d = (65536, 131072] 301133b6
  = (131072, inf) 9433061d
* ``sm_100a`` ``bwd_residual_h6144``: (0, 2] deeb1876 (one cluster) ? (2, 64] 3a55646d =
  (64, 1408] 864e84f3 ? (1408, 65536] f6537c86 ? (65536, 131072] 6feab384 =
  (131072, inf) 3fbfdd75
* ``sm_100a`` ``bwd_w32_h6144``: (0, 4] 35b97e6e (one cluster) ? (4, 64] f4d41a09 =
  (64, 1168] b3e6531e = (1168, 3584] 41b80b6a (det pair) = (3584, 65536] a562656c =
  (65536, 131072] c61519fe = (131072, inf) a562656c
* ``sm_100a`` ``bwd_w32_residual_h6144``: (0, 4] a6b1a6e5 (one cluster) ?
  (4, 64] 6e0d06ab = (64, 1408] 71d36617 = (1408, 65536] a640d0b7 =
  (65536, 131072] f5453c09 = (131072, inf) a640d0b7
* ``sm_103a`` ``bwd_h128``: (0, 1024] 93466008 ? (1024, 2048] 401a018c ?
  (2048, 8192] d197bcc3 (det pair) ? (8192, 12288] 8b0b86c8 ? (12288, 16384] c50d3bbd ?
  (16384, inf) 8b0b86c8
* ``sm_103a`` ``bwd_h256``: (0, 4] 780c64c6 (one cluster) ?
  (4, 24] f2ccf719 (one cluster) ? (24, 64] 9fb6b410 (one cluster) ?
  (64, 1080] a385ae16 (det pair) = (1080, 2048] af005462 (det pair) ?
  (2048, 262144] 6bdbfe61 (det pair) ? (262144, 720896] a385ae16 (det pair) =
  (720896, 1572864] a385ae16 (det pair) = (1572864, inf) a385ae16 (det pair)
* ``sm_103a`` ``bwd_h512``: (0, 4] 64a29969 (one cluster) ?
  (4, 16] c167e0d2 (one cluster) ? (16, 64] bdfd7789 = (64, 1024] 09601d42 (det pair) =
  (1024, 4096] 8e7f4d81 (det pair) = (4096, 32768] 274b0055 (det pair) =
  (32768, inf) cfaae3ce (det pair)
* ``sm_103a`` ``bwd_w32_h512``: (0, 4] 8c261dfc (one cluster) ?
  (4, 16] 88e6252b (one cluster) ? (16, 227] 2c60470d = (227, 1024] 9f07cea1 (det pair)
  = (1024, 32768] 56ee8e66 (det pair) = (32768, inf) a546f410 (det pair)
* ``sm_103a`` ``bwd_h1024``: (0, 4] 330930e9 (one cluster) ? (4, 64] 89fa3e96 ?
  (64, 128] 5ad5f096 ? (128, 1024] 89fa3e96 ? (1024, 1536] 5e32dd88 (det pair) ?
  (1536, 2048] 33d90259 ? (2048, 32768] af798d10 ? (32768, inf) acd331db (det pair)
* ``sm_103a`` ``bwd_w32_h1024``: (0, 4] 81e742d5 (one cluster) ?
  (4, 8] 8501e95e (one cluster) ? (8, 1024] 744d7429 ? (1024, 1536] ba8feeec (det pair)
  ? (1536, 2048] 1a45ade5 ? (2048, 4096] 1a45ade5 ? (4096, 12288] 1a45ade5 ?
  (12288, 32768] 1a45ade5 ? (32768, 131072] 1a45ade5 ? (131072, inf) 2980f513
* ``sm_103a`` ``bwd_h1280``: (0, 1] 8895b022 ? (1, 128] a5f2942d ? (128, 512] 809235a3 ?
  (512, 2048] 12bbedff (det pair) ? (2048, inf) b2cf87e6
* ``sm_103a`` ``bwd_w32_h1280``: (0, 1] 103805d9 ? (1, 128] 8d8ef930 ?
  (128, 512] e476d0b2 ? (512, 4096] 80956d25 (det pair) ? (4096, inf) f6956a41
* ``sm_103a`` ``bwd_h1536``: (0, 4] db218c92 (one cluster) ?
  (4, 5] c2a523de (one cluster) ? (5, 64] 02e761bc = (64, 214] 4aef8d1f =
  (214, 1024] 7dc29629 (det pair) = (1024, 1216] c1a9efdf (det pair) ?
  (1216, 4096] 38e32d90 (det pair) ? (4096, 46592] 38e32d90 (det pair) ?
  (46592, 53184] 20d7a230 ? (53184, inf) 38e32d90 (det pair)
* ``sm_103a`` ``bwd_w32_h1536``: (0, 4] 6d3986ee (one cluster) ?
  (4, 6] 744aa774 (one cluster) ? (6, 64] 166e288c = (64, 214] 698c3d36 =
  (214, 1024] baff9ecb (det pair) = (1024, 1216] 17d9cdfd (det pair) ?
  (1216, 4096] e30c1899 (det pair) ? (4096, 46592] e30c1899 (det pair) ?
  (46592, 53184] 51f46ff1 (det pair) ? (53184, inf) e30c1899 (det pair)
* ``sm_103a`` ``bwd_h2048``: (0, 4] c5f62407 (one cluster) ?
  (4, 16] a951623c (one cluster) ? (16, 63] 518cf16a = (63, 171] ed36a8d9 ?
  (171, 16384] cc26e316 (det pair) ? (16384, 32768] caf6978e (det pair) =
  (32768, 58496] 302f64db (det pair) = (58496, 61952] 302f64db (det pair) =
  (61952, 68160] 302f64db (det pair) = (68160, 131072] 87532fb0 = (131072, inf) 33e5c382
* ``sm_103a`` ``bwd_w32_h2048``: (0, 4] 0017ab41 (one cluster) ?
  (4, 16] 65e5f202 (one cluster) ? (16, 64] c2f08c95 = (64, 171] 978741b4 ?
  (171, 16384] c83c314c (det pair) ? (16384, 32768] 07f9f2fb (det pair) =
  (32768, 58496] e98cb156 (det pair) = (58496, 61952] e98cb156 (det pair) =
  (61952, 68160] e98cb156 (det pair) = (68160, 131072] 149b129c = (131072, inf) 1166d499
* ``sm_103a`` ``bwd_h4096``: (0, 4] 945a0afd (one cluster) ?
  (4, 16] 41f44b4b (one cluster) ? (16, 32] dc7ed206 = (32, 152] 12d53fe5 =
  (152, 256] 50044ee3 (det pair) = (256, 1024] ad2bb26e (det pair) ?
  (1024, 20480] 5572b267 (det pair) ? (20480, 65536] ef9f8d46 ? (65536, 262144] f0aebb74
  ? (262144, inf) 35888fb3
* ``sm_103a`` ``bwd_w32_h4096``: (0, 4] f6bd1634 (one cluster) ?
  (4, 16] b378ba00 (one cluster) ? (16, 32] 8274f36f = (32, 152] 27615d0e =
  (152, 256] 3fd77c8c (det pair) = (256, 1024] 16545056 (det pair) ?
  (1024, 20480] 47a01388 (det pair) ? (20480, 48640] 626aed74 (det pair) ?
  (48640, 65536] a8eb7547 (det pair) ? (65536, 262144] bd22da7e = (262144, inf) 36790f79
* ``sm_103a`` ``bwd_h5120``: (0, 4] dbc680f3 (one cluster) ? (4, 16] 29bcbde1 ?
  (16, 64] 080e0a6c ? (64, 192] 2fc4932f (det pair) ? (192, 1024] bfde9b88 (det pair) ?
  (1024, 36301] bfde9b88 (det pair) ? (36301, 61952] 2fc4932f (det pair) =
  (61952, 393216] e21acade = (393216, inf) 826e6a14
* ``sm_103a`` ``bwd_w32_h5120``: (0, 4] d4713822 (one cluster) ?
  (4, 5] 05396cd3 (one cluster) ? (5, 16] cb653957 ? (16, 192] 139d162a ?
  (192, 1024] f9b2d42c (det pair) ? (1024, 36301] f9b2d42c (det pair) ?
  (36301, 61952] 79811a76 (det pair) = (61952, 196608] 05078a17 = (196608, inf) 91a40f30
* ``sm_103a`` ``bwd_h6144``: (0, 4] e37752b0 (one cluster) ?
  (4, 6] fdfb7335 (one cluster) ? (6, 16] e6475d8f ? (16, 896] 44508df7 ?
  (896, 4096] 61421bc7 (det pair) ? (4096, 24575] 322574af (det pair) ?
  (24575, 65536] 455174e1 ? (65536, 131072] 850fba46 ? (131072, inf) 38337af1
* ``sm_103a`` ``bwd_residual_h6144``: (0, 4] deeb1876 (one cluster) ?
  (4, 6] 94c547ec (one cluster) ? (6, 31] 3a55646d = (31, 32] 864e84f3 =
  (32, 48] 3a55646d = (48, 56] 864e84f3 ? (56, 304] 864e84f3 ? (304, 1920] 864e84f3 ?
  (1920, 131072] aac3ebbf ? (131072, inf) 115dcc71
* ``sm_103a`` ``bwd_w32_h6144``: (0, 4] 35b97e6e (one cluster) ?
  (4, 12] 18b8b805 (one cluster) ? (12, 16] f4d41a09 ? (16, 1024] b3e6531e ?
  (1024, 4096] 41b80b6a (det pair) = (4096, 49152] 9a8c0396 (det pair) =
  (49152, 83392] 9a8c0396 (det pair) = (83392, 131072] cb1aee12 = (131072, inf) 5a635713
* ``sm_103a`` ``bwd_w32_residual_h6144``: (0, 4] a6b1a6e5 (one cluster) ?
  (4, 10] c29a8f85 (one cluster) ? (10, 31] 6e0d06ab = (31, 32] 71d36617 =
  (32, 48] 6e0d06ab ? (48, 320] 71d36617 ? (320, 1824] 71d36617 ?
  (1824, 131072] 5fbcc18d ? (131072, inf) 3c390339
* ``sm_107a`` ``bwd_h128``: (0, 1024] 93466008 ? (1024, 2047] 401a018c ?
  (2047, 65536] d197bcc3 (det pair) ? (65536, inf) 8b0b86c8
* ``sm_107a`` ``bwd_h256``: (0, 4] 780c64c6 (one cluster) ?
  (4, 24] f2ccf719 (one cluster) ? (24, 64] 9fb6b410 (one cluster) ?
  (64, 786432] a385ae16 (det pair) = (786432, inf) 44ef3141
* ``sm_107a`` ``bwd_h512``: (0, 4] 64a29969 (one cluster) ?
  (4, 16] c167e0d2 (one cluster) ? (16, 576] 49af1e75 = (576, 1024] 3feed502 (det pair)
  = (1024, 4096] 3feed502 (det pair) = (4096, 16384] dfaf378a (det pair) =
  (16384, 49152] eb141cc9 (det pair) = (49152, inf) 7764c8a3
* ``sm_107a`` ``bwd_w32_h512``: (0, 4] 8c261dfc (one cluster) ?
  (4, 16] 88e6252b (one cluster) ? (16, 384] 8f29656c = (384, 1024] b841d64b (det pair)
  = (1024, 4096] fd08c6f4 = (4096, 16384] 383f709c (det pair) =
  (16384, 49152] 03efa681 (det pair) = (49152, inf) 5117a0f6
* ``sm_107a`` ``bwd_h1024``: (0, 4] 330930e9 (one cluster) ?
  (4, 8] 28647b7c (one cluster) ? (8, 1536] 89fa3e96 ? (1536, 2048] 0c581a13 (det pair)
  ? (2048, 4095] 5e32dd88 (det pair) ? (4095, inf) 4572a519 (det pair)
* ``sm_107a`` ``bwd_w32_h1024``: (0, 4] 81e742d5 (one cluster) ?
  (4, 16] 8501e95e (one cluster) ? (16, 1536] 744d7429 ?
  (1536, 4095] 561bea88 (det pair) ? (4095, inf) 64423f2d (det pair)
* ``sm_107a`` ``bwd_h1280``: (0, 1] 8895b022 ? (1, 128] a5f2942d ? (128, 512] 809235a3 ?
  (512, 6143] 12bbedff (det pair) ? (6143, 32767] b1339aac (det pair) ?
  (32767, inf) f8cb729b (det pair)
* ``sm_107a`` ``bwd_w32_h1280``: (0, 4] 7905241d (one cluster) ? (4, 128] 8d8ef930 ?
  (128, 512] e476d0b2 ? (512, 4096] 80956d25 (det pair) ?
  (4096, 6143] c66b8cbd (det pair) ? (6143, 32767] de2f75d4 (det pair) ?
  (32767, inf) c5386a35 (det pair)
* ``sm_107a`` ``bwd_h1536``: (0, 4] db218c92 (one cluster) ?
  (4, 16] c2a523de (one cluster) ? (16, 320] 1207b056 = (320, 1024] 7dc29629 (det pair)
  = (1024, 12288] 88510c60 (det pair) = (12288, 16384] 6a428da2 (det pair) =
  (16384, 45056] 56d26996 (det pair) ? (45056, 65536] 411e270c (det pair) ?
  (65536, inf) c93d0266
* ``sm_107a`` ``bwd_w32_h1536``: (0, 4] 6d3986ee (one cluster) ?
  (4, 16] 744aa774 (one cluster) ? (16, 320] 137ba532 = (320, 1024] baff9ecb (det pair)
  = (1024, 16384] 7c0dd63f = (16384, 45056] de8e07d4 (det pair) ?
  (45056, 65536] 64ffe19b (det pair) ? (65536, inf) c9dc9e43
* ``sm_107a`` ``bwd_h2048``: (0, 4] c5f62407 (one cluster) ?
  (4, 16] a951623c (one cluster) ? (16, 31] 518cf16a = (31, 256] 6720b4a1 =
  (256, 1024] 845cdcd8 (det pair) ~ (1024, 16384] cc26e316 (det pair) =
  (16384, 50176] 845cdcd8 (det pair) = (50176, 50688] 395fedeb ?
  (50688, 65536] 21ee6fc9 (det pair) ? (65536, inf) 395fedeb
* ``sm_107a`` ``bwd_w32_h2048``: (0, 4] 0017ab41 (one cluster) ?
  (4, 16] 65e5f202 (one cluster) ? (16, 256] 23c90013 = (256, 1024] 838a9a40 (det pair)
  = (1024, 16384] e90d94fc = (16384, 50176] 838a9a40 (det pair) = (50176, inf) e90d94fc
* ``sm_107a`` ``bwd_h4096``: (0, 4] 945a0afd (one cluster) ?
  (4, 16] 41f44b4b (one cluster) ? (16, 256] c1de7e41 = (256, 1024] a647fccc (det pair)
  = (1024, 4096] d41a514f (det pair) = (4096, 24064] ee4a9053 (det pair) ?
  (24064, 24576] a647fccc (det pair) ? (24576, 65536] 23573e0c ? (65536, inf) f11a8616
* ``sm_107a`` ``bwd_w32_h4096``: (0, 4] f6bd1634 (one cluster) ?
  (4, 16] b378ba00 (one cluster) ? (16, 256] 30ec8bb6 = (256, 1024] dc439f83 (det pair)
  = (1024, 4096] b0863320 (det pair) = (4096, 24064] 0f1859b9 (det pair) ?
  (24064, 65536] 0ff440da (det pair) ? (65536, inf) a1f537fc
* ``sm_107a`` ``bwd_h5120``: (0, 4] dbc680f3 (one cluster) ?
  (4, 12] 904b1ad4 (one cluster) ? (12, 31] 29bcbde1 = (31, 256] 080e0a6c =
  (256, 511] 838c077d (det pair) = (511, 640] 080e0a6c = (640, 895] 838c077d (det pair)
  = (895, 896] 080e0a6c = (896, 4096] 838c077d (det pair) =
  (4096, 19968] 6816fb80 (det pair) = (19968, 49152] b8b0913b (det pair) =
  (49152, inf) 670f4caf
* ``sm_107a`` ``bwd_w32_h5120``: (0, 4] d4713822 (one cluster) ?
  (4, 15] 05396cd3 (one cluster) ? (15, 256] 139d162a = (256, 511] cd612de5 (det pair) =
  (511, 640] 139d162a = (640, 895] cd612de5 (det pair) = (895, 1280] 139d162a =
  (1280, 4096] cd612de5 (det pair) ? (4096, 19968] 468cc4d1 (det pair) ?
  (19968, 32767] de0cf457 (det pair) ? (32767, 131072] e7cb6972 ? (131072, inf) 13bd0c5f
* ``sm_107a`` ``bwd_h6144``: (0, 4] e37752b0 (one cluster) ?
  (4, 16] fdfb7335 (one cluster) ? (16, 960] 0dafdf09 = (960, 1151] 5d292172 (det pair)
  = (1151, 1152] 0dafdf09 = (1152, 19456] 5d292172 (det pair) =
  (19456, 30720] 9cc51adf (det pair) ? (30720, 131072] ffbcfda2 ? (131072, inf) ddedaa3b
* ``sm_107a`` ``bwd_residual_h6144``: (0, 4] deeb1876 (one cluster) ?
  (4, 15] 94c547ec (one cluster) ? (15, 320] ee1be07a = (320, 447] 0dcdc563 (det pair) =
  (447, 576] ee1be07a = (576, 3584] 864e84f3 = (3584, 7168] 0dcdc563 (det pair) ?
  (7168, 131072] 853d4447 ? (131072, inf) 87d2866c
* ``sm_107a`` ``bwd_w32_h6144``: (0, 4] 35b97e6e (one cluster) ?
  (4, 16] 18b8b805 (one cluster) ? (16, 1664] 5cca2d0f = (1664, 2304] b3e6531e =
  (2304, 19456] e77387f1 = (19456, 28672] bdfba721 (det pair) = (28672, inf) e77387f1
* ``sm_107a`` ``bwd_w32_residual_h6144``: (0, 4] a6b1a6e5 (one cluster) ?
  (4, 16] c29a8f85 (one cluster) ? (16, 768] 3bbd4f84 = (768, 3584] 71d36617 =
  (3584, 7168] 56f988dd (det pair) ? (7168, 131072] b394bc75 ? (131072, inf) 64758944
"""

SECTIONS: dict[str, str] = {"fwd": FORWARD_SECTION, "bwd": BACKWARD_SECTION}

__all__ = [
    "BACKWARD_SECTION",
    "DET_PAIR_ROUTES",
    "EDGES",
    "EDGE_CLASSES",
    "FORWARD_SECTION",
    "NOTES",
    "PROGRAM_WINDOWS",
    "SECTIONS",
    "SMEM_SIDE_KB",
    "SMEM_WAITS",
    "WHOLE_SM",
    "WHOLE_SM_REMEDY",
]
