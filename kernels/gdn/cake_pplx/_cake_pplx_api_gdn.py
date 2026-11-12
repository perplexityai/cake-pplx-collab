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

It does nothing: ``@cake_api`` and ``@cake_api(...)`` both return the
function unchanged.  The module also provides no-op custom-op registration
wrappers and API-trace dispatchers for the facades.  Each family carries its own copy; no family imports another's.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["cake_api", "register_custom_op", "register_fake_op"]


def cake_api(function: Callable[..., Any] | None = None, **_options: Any) -> Any:
    if function is None:
        return lambda inner: inner
    return function


def register_custom_op(name: str, fn: Callable[..., Any] | None = None, /, **_options: Any) -> Any:
    """Registers nothing: returns the function unchanged."""
    return lambda inner: inner


def register_fake_op(name: str, fn: Callable[..., Any] | None = None) -> Any:
    """Registers nothing: returns the function unchanged."""
    return lambda inner: inner


def __getattr__(name: str) -> Any:
    # API-trace dispatchers (``<entry>_trace_dispatch``); ``cake_api`` ignores them.
    if name.endswith("_trace_dispatch"):
        return None
    raise AttributeError(name)
