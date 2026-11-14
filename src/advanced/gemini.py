"""Minimal Gemini API client (standard library only) used by the advanced challenge.

The key is read from the environment variable GEMINI_API_KEY or from a `.env` file at the
repo root (git-ignored). The key is never printed or written to traces.
Rate-limit (429) and transient server errors are retried with backoff; other errors raise.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
API = 'https://generativelanguage.googleapis.com/v1beta'


def api_key() -> str:
    key = os.environ.get('GEMINI_API_KEY', '').strip()
    env = ROOT / '.env'
    if not key and env.exists():
        for line in env.read_text(encoding='utf-8').splitlines():
            name, _, value = line.partition('=')
            if name.strip() == 'GEMINI_API_KEY':
                key = value.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError('Chưa có GEMINI_API_KEY trong biến môi trường hoặc file .env')
    return key


def default_model() -> str:
    return os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')


class DailyQuotaExceeded(RuntimeError):
    """The free-tier daily request quota of this model is used up; retrying today is pointless."""


def upper_types(schema: Any) -> Any:
    """Gemini's responseSchema uses upper-case OpenAPI type names."""
    if isinstance(schema, dict):
        return {k: (v.upper() if k == 'type' and isinstance(v, str) else upper_types(v)) for k, v in schema.items()}
    if isinstance(schema, list):
        return [upper_types(x) for x in schema]
    return schema


def image_part(path: Path) -> dict[str, Any]:
    return {'inline_data': {'mime_type': 'image/png',
                            'data': base64.b64encode(Path(path).read_bytes()).decode('ascii')}}


def _request(url: str, body: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    data = json.dumps(body).encode('utf-8') if body is not None else None
    request = urllib.request.Request(url, data=data, headers={
        'Content-Type': 'application/json', 'x-goog-api-key': api_key()})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def list_models() -> list[str]:
    out = _request(f'{API}/models?pageSize=200', None, 60)
    return [m['name'].removeprefix('models/') for m in out.get('models', [])
            if 'generateContent' in m.get('supportedGenerationMethods', [])]


def generate(contents: list[dict[str, Any]], *, model: str | None = None, system: str | None = None,
             tools: list[dict[str, Any]] | None = None, schema: dict[str, Any] | None = None,
             temperature: float = 0.0, max_retries: int = 8, timeout: int = 300) -> dict[str, Any]:
    """One generateContent call. Returns {'text', 'function_calls', 'usage', 'seconds', 'model', 'retries'}."""
    model = model or default_model()
    body: dict[str, Any] = {'contents': contents, 'generationConfig': {'temperature': temperature}}
    if system:
        body['systemInstruction'] = {'parts': [{'text': system}]}
    if tools:
        body['tools'] = [{'functionDeclarations': upper_types(tools)}]
    if schema:
        body['generationConfig'].update({'responseMimeType': 'application/json', 'responseSchema': upper_types(schema)})
    started, retries, errors = time.perf_counter(), 0, []
    while True:
        try:
            out = _request(f'{API}/models/{model}:generateContent', body, timeout)
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', 'replace')[:2000]
            if error.code == 429 and 'PerDay' in detail:
                raise DailyQuotaExceeded(f'{model}: daily quota exceeded') from None
            if error.code in (429, 500, 502, 503, 504) and retries < max_retries:
                retries += 1
                errors.append(error.code)
                print(f'  gemini HTTP {error.code}, retry {retries}: {detail[:160]!r}', flush=True)
                time.sleep(min(40, 4 * 2 ** retries))
                continue
            raise RuntimeError(f'Gemini HTTP {error.code}: {detail}') from None
        except (urllib.error.URLError, TimeoutError) as error:
            if retries < max_retries:
                retries += 1
                time.sleep(min(90, 5 * 2 ** retries))
                continue
            raise RuntimeError(f'Gemini connection error: {error}') from None
    candidate = (out.get('candidates') or [{}])[0]
    parts = candidate.get('content', {}).get('parts', [])
    return {'text': ''.join(p.get('text', '') for p in parts if not p.get('thought')),
            'function_calls': [p['functionCall'] for p in parts if 'functionCall' in p],
            'finish_reason': candidate.get('finishReason'),
            'usage': out.get('usageMetadata', {}), 'seconds': time.perf_counter() - started,
            'model': model, 'retries': retries, 'retry_errors': errors}


if __name__ == '__main__':
    print('\n'.join(m for m in list_models() if 'gemini' in m))
