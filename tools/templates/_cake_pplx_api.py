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

The public-entry decorator of one ``cake_pplx`` family.

It replaces FlashInfer's ``flashinfer_api`` logging decorator and does nothing:
``@cake_api`` and ``@cake_api(...)`` both return the function unchanged.  Each
family carries its own copy; no family imports another's.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["cake_api"]


def cake_api(function: Callable[..., Any] | None = None, **_options: Any) -> Any:
    if function is None:
        return lambda inner: inner
    return function
