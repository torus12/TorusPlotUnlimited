"""AI providers. Backends are imported lazily by :func:`get_provider`.

Importing this package pulls in no network code and no SDK: only the base interface and
the offline mock provider.
"""

from .base import Provider, ProviderError, active_provider, get_provider, image_media_type
from .mock import MockProvider

__all__ = [
    "MockProvider",
    "Provider",
    "ProviderError",
    "active_provider",
    "get_provider",
    "image_media_type",
]
