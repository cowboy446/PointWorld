#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple Hugging Face input/cache helpers for BEHAVIOR raw episodes."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

POINTWORLD_CACHE_ENV = "POINTWORLD_CACHE_DIR"
HF_TOKEN_ENV = "HF_TOKEN"
HF_ENDPOINT_ENV = "HF_ENDPOINT"
HF_DOWNLOAD_RETRIES_ENV = "POINTWORLD_HF_DOWNLOAD_RETRIES"
BEHAVIOR_HF_ORG = "behavior-1k"
BEHAVIOR_HF_REPO = "2025-challenge-rawdata"

HF_DATASET_URL_RE = re.compile(
    r"^https?://[^/]+/datasets/(?P<org>[^/]+)/(?P<repo>[^/]+)/(?P<mode>resolve|blob|raw)/(?P<rev>[^/]+)/(?P<file>.+)$"
)
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def is_hf_path(path: str) -> bool:
    """Return True if the path points to a Hugging Face dataset file."""
    if path.startswith("hf://"):
        return True
    return HF_DATASET_URL_RE.match(path) is not None


def is_behavior_hf_path(path: str) -> bool:
    """Return True if path points to the canonical BEHAVIOR raw HF dataset."""
    try:
        org, repo, _revision, _file_path = _extract_hf_parts(path)
    except Exception:
        return False
    return org == BEHAVIOR_HF_ORG and repo == BEHAVIOR_HF_REPO


def hf_input_to_resolve_url(path: str) -> str:
    """Normalize hf:// shorthand or dataset URL to a resolve URL."""
    if path.startswith("hf://"):
        # hf://org/repo/path/to/file.hdf5 or hf://org/repo@revision/path/to/file.hdf5
        raw = path[len("hf://") :]
        parts = [p for p in raw.split("/") if p]
        if len(parts) < 3:
            raise ValueError(
                "Invalid hf:// input path. Expected hf://org/repo/path/to/file.hdf5 "
                "or hf://org/repo@revision/path/to/file.hdf5"
            )
        org = parts[0]
        repo_part = parts[1]
        file_path = "/".join(parts[2:])
        if "@" in repo_part:
            repo, revision = repo_part.rsplit("@", maxsplit=1)
        else:
            repo, revision = repo_part, "main"
        return _build_hf_resolve_url(org=org, repo=repo, revision=revision, file_path=file_path)

    match = HF_DATASET_URL_RE.match(path)
    if match is None:
        raise ValueError(f"Not a supported Hugging Face dataset path: {path}")

    info = match.groupdict()
    return _build_hf_resolve_url(
        org=info["org"],
        repo=info["repo"],
        revision=info["rev"],
        file_path=info["file"],
    )


def derive_behavior_output_relpath(input_path: str, local_input_root: str | None) -> str:
    """Derive output .h5 relative path for local or HF-backed input."""
    if not is_hf_path(input_path):
        if local_input_root is None:
            rel = os.path.basename(input_path)
        else:
            rel = os.path.relpath(input_path, local_input_root)
        return os.path.splitext(rel)[0] + ".h5"

    org, repo, revision, file_path = _extract_hf_parts(input_path)
    repo_safe = f"{org}__{repo}"
    return os.path.join("hf", repo_safe, revision, os.path.splitext(file_path)[0] + ".h5")


def get_local_behavior_input_path(input_path: str, temp_dir: str | None = None) -> str:
    """Return a local file path for a BEHAVIOR input path."""
    if not is_hf_path(input_path):
        return input_path

    resolve_url = hf_input_to_resolve_url(input_path)
    org, repo, revision, file_path = _extract_hf_parts(resolve_url)

    cache_root = os.environ.get(POINTWORLD_CACHE_ENV, "").strip()
    token = os.environ.get(HF_TOKEN_ENV, "").strip() or None

    if cache_root:
        local_path = os.path.join(cache_root, "behavior", f"{org}__{repo}", revision, file_path)
        if os.path.isfile(local_path):
            if not _has_hdf5_signature(local_path):
                print(f"[behavior raw cache] removing invalid cached HDF5: {local_path}")
                os.remove(local_path)
            else:
                return local_path
        if not os.path.isfile(local_path):
            _download_hf_file(resolve_url, local_path, token)
            _validate_hdf5_signature(local_path, resolve_url)
            return local_path

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="behavior_raw_stream_")
    local_path = os.path.join(temp_dir, os.path.basename(file_path))
    if not os.path.isfile(local_path):
        _download_hf_file(resolve_url, local_path, token)
        _validate_hdf5_signature(local_path, resolve_url)
    return local_path


def enforce_behavior_cache_policy(
    input_paths,
    stage_name: str,
    require_cache: bool = False,
    allow_streaming: bool = False,
) -> bool:
    """Validate POINTWORLD cache policy for HF-backed BEHAVIOR inputs."""
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    remote_paths = [p for p in input_paths if is_hf_path(p)]
    if not remote_paths:
        return False

    cache_root = os.environ.get(POINTWORLD_CACHE_ENV, "").strip()
    if cache_root:
        print(f"[{stage_name}] cache enabled: {POINTWORLD_CACHE_ENV}={cache_root}")
        return True

    message = (
        f"[{stage_name}] Detected {len(remote_paths)} Hugging Face input path(s) but {POINTWORLD_CACHE_ENV} is not set. "
        "Without a persistent cache, repeated stages/reruns may re-download the same files."
    )
    setup_hint = (
        "Set a shared cache root before running this stage, for example: "
        f"export {POINTWORLD_CACHE_ENV}=/path/to/fast_disk/pointworld_cache"
    )

    if require_cache and not allow_streaming:
        raise RuntimeError(
            f"{message} {setup_hint} "
            "If you intentionally want to stream again, rerun with --allow_remote_streaming."
        )

    if allow_streaming:
        print(f"WARNING: {message} Continuing because --allow_remote_streaming was provided.")
    else:
        print(f"WARNING: {message} {setup_hint}")

    return True


def _build_hf_resolve_url(org: str, repo: str, revision: str, file_path: str) -> str:
    org_q = quote(org, safe="")
    repo_q = quote(repo, safe="")
    revision_q = quote(revision, safe="")
    file_q = quote(file_path, safe="/")
    endpoint = os.environ.get(HF_ENDPOINT_ENV, "https://huggingface.co").strip().rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"{HF_ENDPOINT_ENV} must be an HTTP(S) URL, got: {endpoint}")
    return f"{endpoint}/datasets/{org_q}/{repo_q}/resolve/{revision_q}/{file_q}"


def _extract_hf_parts(path: str) -> tuple[str, str, str, str]:
    resolve_url = hf_input_to_resolve_url(path)
    match = HF_DATASET_URL_RE.match(resolve_url)
    if match is None:
        raise ValueError(f"Could not parse Hugging Face dataset URL: {path}")
    info = match.groupdict()
    return info["org"], info["repo"], info["rev"], info["file"]


def _has_hdf5_signature(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(HDF5_MAGIC)) == HDF5_MAGIC
    except OSError:
        return False


def _validate_hdf5_signature(path: str, source_url: str) -> None:
    if _has_hdf5_signature(path):
        return
    if os.path.exists(path):
        os.remove(path)
    raise RuntimeError(
        "Downloaded BEHAVIOR raw episode is not a valid HDF5 file. "
        f"Removed invalid file: {path}. Source: {source_url}"
    )


def _download_hf_file(url: str, local_path: str, token: str | None) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = f"{local_path}.tmp"

    headers = {"User-Agent": "pointworld-behavior/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        max_attempts = max(1, int(os.environ.get(HF_DOWNLOAD_RETRIES_ENV, "8")))
    except ValueError as exc:
        raise ValueError(f"{HF_DOWNLOAD_RETRIES_ENV} must be an integer") from exc

    last_error = None
    for attempt in range(1, max_attempts + 1):
        downloaded = os.path.getsize(tmp_path) if os.path.isfile(tmp_path) else 0
        request_headers = dict(headers)
        if downloaded:
            request_headers["Range"] = f"bytes={downloaded}-"
            print(f"[behavior raw cache] resuming at byte {downloaded}: {url}")

        try:
            request = Request(url, headers=request_headers)
            with urlopen(request, timeout=300) as response:
                # A server may ignore Range and return the whole file with 200.
                append = downloaded > 0 and getattr(response, "status", None) == 206
                with open(tmp_path, "ab" if append else "wb") as out_f:
                    shutil.copyfileobj(response, out_f)
            os.replace(tmp_path, local_path)
            return
        except HTTPError as exc:
            last_error = exc
            if exc.code in {401, 403}:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise RuntimeError(
                    f"Hugging Face download failed ({exc.code}) for {url}. "
                    "If this file is gated/private, set HF_TOKEN with read access."
                ) from exc
            if exc.code == 416 and os.path.exists(tmp_path):
                # The partial file may be stale or already complete; restart cleanly.
                os.remove(tmp_path)
            if exc.code not in {408, 416, 429} and not 500 <= exc.code < 600:
                raise RuntimeError(f"HTTP error while downloading {url}: {exc}") from exc
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc

        if attempt < max_attempts:
            delay = min(60, 2 ** (attempt - 1))
            print(
                f"[behavior raw cache] download attempt {attempt}/{max_attempts} failed: "
                f"{last_error}; retrying in {delay}s"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Download failed after {max_attempts} attempts for {url}: {last_error}"
    ) from last_error
