"""
scripts/_env.py -- Load and validate NASA Earthdata bearer token from .env.

Imported by fetch_nisar.py and train.py. Not a runnable script.

Setup
-----
1. Copy .env.example to .env in the project root
2. Go to https://urs.earthdata.nasa.gov/profile -> click "Generate Token"
3. Paste the token into .env:
       EARTHDATA_TOKEN=eyJ0eXAi...

Tokens expire after 60 days. Generate a new one at the same URL and update .env.

Common reasons auth_with_token fails even with a valid token
-------------------------------------------------------------
1. Copied the wrong thing -- the token is the long string starting with 'eyJ',
   NOT the curl example, NOT the JSON wrapper, just the token value itself.
2. Whitespace or quotes around the token in .env.
3. Token not yet associated with the ASF application -- visit
   https://search.asf.alaska.edu and sign in once with your EDL credentials.
   This authorises the ASF app on your account. Required before token auth works.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_token(env_file: Path | None = None) -> str:
    """
    Load and validate the Earthdata bearer token.

    Performs basic sanity checks before returning so that errors are caught
    locally with a clear message rather than inside asf_search internals.

    Parameters
    ----------
    env_file : path to .env file; defaults to <project_root>/.env

    Returns
    -------
    token : str -- raw EDL bearer token, stripped of whitespace and quotes

    Raises
    ------
    RuntimeError if token is missing or fails basic format validation.
    """
    if env_file is None:
        env_file = Path(__file__).resolve().parent.parent / '.env'

    if env_file.is_file():
        _load_dotenv(env_file)
    else:
        print(f"  Warning: .env not found at {env_file}")

    raw = os.environ.get('EARTHDATA_TOKEN', '')

    # Strip whitespace, newlines, and any accidental surrounding quotes
    token = raw.strip().strip('"').strip("'").strip()

    if not token:
        raise RuntimeError(
            "EARTHDATA_TOKEN is empty or not set.\n\n"
            "Steps:\n"
            "  1. Go to https://urs.earthdata.nasa.gov/profile\n"
            "  2. Click 'Generate Token'\n"
            "  3. Copy the token string (starts with 'eyJ')\n"
            "  4. Paste into .env as:  EARTHDATA_TOKEN=eyJ0eXAi...\n\n"
            f"Expected .env location: {env_file}"
        )

    # EDL tokens are JWTs -- they must start with 'eyJ' (base64 of '{"')
    if not token.startswith('eyJ'):
        raise RuntimeError(
            f"EARTHDATA_TOKEN looks wrong -- got '{token[:20]}...'\n\n"
            "A valid EDL token starts with 'eyJ'. Common mistakes:\n"
            "  - Copied the JSON wrapper instead of just the token value\n"
            "  - Copied the expiry date or the curl command instead\n"
            "  - Extra characters before the token in .env\n\n"
            "The token value to copy is labelled 'access_token' and looks like:\n"
            "  eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJlZGwiOiJ..."
        )

    # EDL JWTs have exactly three dot-separated segments
    if token.count('.') != 2:
        raise RuntimeError(
            f"EARTHDATA_TOKEN has {token.count('.')} dots; expected 2.\n"
            "EDL tokens are JWTs with the format: header.payload.signature\n"
            "Make sure the full token was copied without truncation."
        )

    print(f"  Token: {len(token)} chars  [{token[:8]}...{token[-6:]}]")
    return token


def make_asf_session(env_file: Path | None = None):
    """
    Return an authenticated asf_search.ASFSession using the bearer token.

    IMPORTANT: Before token auth will work, you must visit
    https://search.asf.alaska.edu and sign in with your EDL credentials
    at least once. This authorises the ASF application on your Earthdata
    account. Token auth silently fails until this step is done.

    Parameters
    ----------
    env_file : optional explicit path to .env

    Returns
    -------
    asf_search.ASFSession
    """
    try:
        import asf_search as asf
    except ImportError as exc:
        raise ImportError("Run: pip install asf_search") from exc

    token = load_token(env_file)
    print("  Authenticating with EDL bearer token ...")
    return asf.ASFSession().auth_with_token(token)


# ---------------------------------------------------------------------------
# .env file parser
# ---------------------------------------------------------------------------

def _load_dotenv(env_file: Path) -> None:
    """
    Parse KEY=VALUE lines from a .env file and inject into os.environ.
    Uses python-dotenv if installed; falls back to a built-in parser.
    Does NOT override variables already set in the environment.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file, override=False)
        return
    except ImportError:
        pass

    # Built-in parser -- handles KEY=VALUE and KEY="VALUE"
    with open(env_file, 'r', encoding='utf-8') as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value