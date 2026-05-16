from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener


def patch_hf_cache_home() -> None:
    """兼容"""
    try:
        import huggingface_hub as hh
        import huggingface_hub.constants as hh_constants
    except Exception:
        return

    cache_dir = getattr(hh_constants, "HUGGINGFACE_HUB_CACHE", None)
    if cache_dir:
        cache_home = str(Path(cache_dir).expanduser().resolve().parent)
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_home = str(Path(hf_home).expanduser().resolve())
        else:
            cache_home = str(Path.home() / ".cache" / "huggingface")

    if not hasattr(hh_constants, "hf_cache_home"):
        setattr(hh_constants, "hf_cache_home", cache_home)

    token_path = Path(cache_home) / "token"

    if not hasattr(hh, "HfFolder"):
        class HfFolder:
            path_token = str(token_path)

            @classmethod
            def save_token(cls, token: str) -> None:
                path = Path(cls.path_token)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(token, encoding="utf-8")

            @classmethod
            def get_token(cls) -> str | None:
                get_token = getattr(hh, "get_token", None)
                if callable(get_token):
                    token = get_token()
                    if token:
                        return token
                path = Path(cls.path_token)
                if path.exists():
                    content = path.read_text(encoding="utf-8").strip()
                    return content or None
                return None

            @classmethod
            def delete_token(cls) -> None:
                path = Path(cls.path_token)
                if path.exists():
                    path.unlink()

        setattr(hh, "HfFolder", HfFolder)

    if not hasattr(hh, "cached_download"):
        def cached_download(
            url: str,
            cache_dir: str | os.PathLike | None = None,
            force_download: bool = False,
            proxies: dict | None = None,
            resume_download=None,
            local_files_only: bool = False,
            token=None,
            **kwargs,
        ) -> str:
            del resume_download, kwargs
            if cache_dir:
                base_cache = Path(cache_dir)
            else:
                base_cache = Path(cache_home) / "hub"
            download_dir = base_cache / "legacy_cached_downloads"
            download_dir.mkdir(parents=True, exist_ok=True)

            suffix = Path(url.split("?")[0]).suffix
            filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + suffix
            dst = download_dir / filename

            if dst.exists() and not force_download:
                return str(dst)
            if local_files_only:
                raise FileNotFoundError(f"Local file not found in cache: {dst}")

            headers = {}
            if isinstance(token, str) and token:
                headers["Authorization"] = f"Bearer {token}"
            req = Request(url, headers=headers)
            if proxies:
                opener = build_opener(ProxyHandler(proxies))
                with opener.open(req) as response, open(dst, "wb") as f:
                    f.write(response.read())
            else:
                with build_opener().open(req) as response, open(dst, "wb") as f:
                    f.write(response.read())
            return str(dst)

        setattr(hh, "cached_download", cached_download)


def import_sana_pipeline() -> Any:
    """导入"""
    patch_hf_cache_home()

    try:
        from diffusers import SanaPipeline

        return SanaPipeline
    except ImportError:
        pass

    try:
        from diffusers.pipelines.sana.pipeline_sana import SanaPipeline

        return SanaPipeline
    except Exception:
        pass

    try:
        import diffusers

        version = getattr(diffusers, "__version__", "unknown")
    except Exception:
        version = "unknown"

    raise ImportError(
        "The current diffusers installation does not provide SanaPipeline. "
        f"Detected diffusers version: {version}. "
        "SanaPipeline is available in diffusers>=0.32.0; please upgrade diffusers."
    )
