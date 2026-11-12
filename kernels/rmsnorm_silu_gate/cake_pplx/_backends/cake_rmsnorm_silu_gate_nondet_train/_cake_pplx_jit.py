"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

JIT build shim of one ``cake_pplx`` backend package.

It provides the three JIT names the package uses --
``gen_jit_spec``, ``get_cuda_version`` and ``get_cuda_path`` -- with the same
call shapes, so ``cake_build.py`` and the backend modules keep their logic.
Every backend package carries its own copy; no copy imports another.

Compilation uses these non-debug defaults (``-std=c++17 -O3
-DNDEBUG --compiler-options=-fPIC --expt-relaxed-constexpr
-Xfatbin=-compress-all --compress-mode=size``, plus
``-static-global-template-stub=false`` from CUDA 12.8) and takes the target
architectures only from the caller's ``-gencode`` flags: it never adds a
generic ``sm_XX`` target.  Libraries are cached under
``$CAKE_PPLX_CACHE_DIR`` (default ``~/.cache/cake_pplx``), keyed by name, a
digest of the sources and every flag, and the toolkit version.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import os
import re
import shutil
import subprocess
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

__all__ = ["CudaVersion", "JitSpec", "gen_jit_spec", "get_cuda_path", "get_cuda_version"]

_DEFAULT_CUDA_FLAGS = [
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "--compiler-options=-fPIC",
    "--expt-relaxed-constexpr",
    "-Xfatbin=-compress-all",
    "--compress-mode=size",
]
_DEFAULT_CXX_FLAGS = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-Wno-switch-bool"]


@functools.cache
def get_cuda_path() -> str:
    """The CUDA toolkit root: ``$CUDA_HOME``, ``$CUDA_PATH``, the ``nvcc`` on ``PATH`` or ``/usr/local/cuda``."""
    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value:
            return value
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return str(Path(nvcc).resolve().parent.parent)
    if Path("/usr/local/cuda/bin/nvcc").is_file():
        return "/usr/local/cuda"
    raise RuntimeError("no CUDA toolkit found: set CUDA_HOME or put nvcc on PATH")


@dataclass(frozen=True, order=True)
class CudaVersion:
    """``major.minor[.patch]`` of the local ``nvcc``."""

    major: int
    minor: int
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@functools.cache
def get_cuda_version() -> CudaVersion:
    """The release of ``$CUDA_HOME/bin/nvcc`` (``V13.0.88`` -> ``CudaVersion(13, 0, 88)``)."""
    nvcc = Path(get_cuda_path(), "bin", "nvcc")
    if not nvcc.is_file():
        raise RuntimeError(f"{nvcc} does not exist")
    text = subprocess.run(
        [str(nvcc), "--version"], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    match = re.search(r"\bV(\d+)\.(\d+)\.(\d+)\b", text) or re.search(
        r"release (\d+)\.(\d+)", text
    )
    if match is None:
        raise RuntimeError(f"cannot read the CUDA version from `{nvcc} --version`")
    return CudaVersion(*(int(part) for part in match.groups()))


def _cache_root() -> Path:
    return Path(os.environ.get("CAKE_PPLX_CACHE_DIR", "~/.cache/cake_pplx")).expanduser()


def _system_includes(cuda_home: str) -> list[str]:
    import tvm_ffi.libinfo

    includes = [
        sysconfig.get_path("include"),
        tvm_ffi.libinfo.find_include_path(),
        tvm_ffi.libinfo.find_dlpack_include_path(),
    ]
    if cuda_home != "/usr":
        includes.insert(1, str(Path(cuda_home, "include")))
    return includes


@dataclass
class JitSpec:
    """One shared library to compile from CUDA / C++ sources (``.build()``, ``.build_and_load()``)."""

    name: str
    sources: list[Path]
    extra_cflags: list[str]
    extra_cuda_cflags: list[str]
    extra_ldflags: list[str]
    extra_include_paths: list[Path]
    use_fast_math: bool

    def _commands(self) -> tuple[list[tuple[Path, list[str]]], list[str], Path]:
        cuda_home = get_cuda_path()
        version = get_cuda_version()
        includes = [f"-I{Path(p).resolve()}" for p in self.extra_include_paths]
        # nvcc takes the -isystem path as a separate argument ("-isystem<path>" is an
        # unknown option to nvcc 13).
        for p in _system_includes(cuda_home):
            includes += ["-isystem", str(p)]
        cuda_flags = list(_DEFAULT_CUDA_FLAGS)
        if (version.major, version.minor) >= (12, 8):
            cuda_flags.append("-static-global-template-stub=false")
        if self.use_fast_math:
            cuda_flags.append("-use_fast_math")
        cuda_flags += self.extra_cuda_cflags
        if not any(flag.startswith("-gencode") or flag.startswith("-arch") for flag in cuda_flags):
            raise RuntimeError(f"{self.name}: the JIT spec names no -gencode target")
        cxx_flags = list(_DEFAULT_CXX_FLAGS) + self.extra_cflags
        nvcc = os.environ.get("CAKE_PPLX_NVCC", str(Path(cuda_home, "bin", "nvcc")))
        cxx = os.environ.get("CXX", "c++")
        compiles = []
        for index, source in enumerate(self.sources):
            if source.suffix == ".cu":
                command = [nvcc, *cuda_flags, *includes, "-c", str(source)]
            else:
                command = [cxx, *cxx_flags, *includes, "-c", str(source)]
            compiles.append((Path(f"{index}_{source.stem}.o"), command))
        link = [
            "-shared",
            f"-L{Path(cuda_home, 'lib64')}",
            f"-L{Path(cuda_home, 'lib64', 'stubs')}",
            "-lcudart",
            "-lcuda",
            *[flag for flag in self.extra_ldflags if flag != "-shared"],
        ]
        digest = hashlib.sha256()
        digest.update(str(version).encode())
        for source in self.sources:
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
        for _, command in compiles:
            digest.update("\0".join(command[1:]).encode())
        digest.update("\0".join(link).encode())
        directory = _cache_root() / f"{self.name}_{digest.hexdigest()[:16]}"
        return compiles, [cxx, *link], directory

    def get_library_path(self) -> Path:
        _, _, directory = self._commands()
        return directory / f"{self.name}.so"

    def build(self) -> Path:
        """Compile and link the library once; later calls return the cached file."""
        if os.environ.get("CAKE_PPLX_DISABLE_JIT"):
            raise RuntimeError(f"{self.name}: JIT is disabled and no prebuilt library exists")
        compiles, link, directory = self._commands()
        library = directory / f"{self.name}.so"
        if library.is_file():
            return library
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if library.is_file():
                return library

            def compile_one(item: tuple[Path, list[str]]) -> Path:
                obj, command = item
                target = directory / obj
                result = subprocess.run(
                    [*command, "-o", str(target)], capture_output=True, text=True
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{self.name}: compiling {command[-1]} failed:\n{result.stderr[-8000:]}"
                    )
                return target

            workers = int(os.environ.get("CAKE_PPLX_JIT_WORKERS", str(min(8, os.cpu_count() or 1))))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                objects = list(pool.map(compile_one, compiles))
            partial = directory / f"{self.name}.so.tmp{os.getpid()}"
            result = subprocess.run(
                [link[0], *map(str, objects), *link[1:], "-o", str(partial)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"{self.name}: link failed:\n{result.stderr[-8000:]}")
            os.replace(partial, library)
        return library

    def build_and_load(self) -> Any:
        import tvm_ffi

        return tvm_ffi.load_module(str(self.build()))


def gen_jit_spec(
    name: str,
    sources: Sequence[Union[str, Path]],
    extra_cflags: Optional[list[str]] = None,
    extra_cuda_cflags: Optional[list[str]] = None,
    extra_ldflags: Optional[list[str]] = None,
    extra_include_paths: Optional[list[Union[str, Path]]] = None,
    use_fast_math: bool = True,
    **unsupported: Any,
) -> JitSpec:
    """A :class:`JitSpec` from the ``gen_jit_spec`` keyword arguments (fast math on by default)."""
    if unsupported:
        raise TypeError(f"{name}: unsupported JIT options {sorted(unsupported)}")
    return JitSpec(
        name=name,
        sources=[Path(source).resolve() for source in sources],
        extra_cflags=list(extra_cflags or []),
        extra_cuda_cflags=list(extra_cuda_cflags or []),
        extra_ldflags=list(extra_ldflags or []),
        extra_include_paths=[Path(path) for path in extra_include_paths or []],
        use_fast_math=use_fast_math,
    )
