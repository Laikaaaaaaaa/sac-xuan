"""
SẮC XUÂN VĂN HỌC - Flask Application
Nền tảng AI kết nối di sản văn học Việt Nam
"""

from flask import Flask, render_template, jsonify, request, session, redirect
import os
import json
import time
import base64
import re
import uuid
from pathlib import Path

_DOTENV_PATH = Path(__file__).resolve().parent / '.env'
_DOTENV_MTIME: float | None = None
_reload_dotenv_if_changed = None

# Load local .env if present (keeps secrets out of code)
try:
    from dotenv import load_dotenv  # type: ignore

    def _reload_dotenv_if_changed():
        """Reload .env when file changes (dev convenience)."""
        global _DOTENV_MTIME
        try:
            if _DOTENV_PATH.exists():
                mtime = float(_DOTENV_PATH.stat().st_mtime)
                if _DOTENV_MTIME is None or mtime != _DOTENV_MTIME:
                    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
                    _DOTENV_MTIME = mtime
            else:
                load_dotenv(override=True)
        except Exception:
            pass

    # Initial load
    _reload_dotenv_if_changed()
except Exception:
    pass

from flask_compress import Compress

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sac-xuan-van-hoc-2026'

# Static caching (helps repeat visits) and response compression (helps first load)
# NOTE: In local debug, disable caching so CSS/JS changes show up immediately.
ONE_YEAR_SECONDS = 60 * 60 * 24 * 365
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = ONE_YEAR_SECONDS
app.config['COMPRESS_MIMETYPES'] = [
    'text/html',
    'text/css',
    'application/javascript',
    'application/json',
    'application/octet-stream',
    'model/gltf-binary',
]
app.config['COMPRESS_LEVEL'] = 6
app.config['COMPRESS_MIN_SIZE'] = 512
Compress(app)


@app.after_request
def _dev_no_cache(response):
    """Avoid stale assets during development (browser may cache aggressively)."""
    if app.debug:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.before_request
def _dev_reload_env():
    """In debug, reload .env automatically when edited."""
    if app.debug and callable(_reload_dotenv_if_changed):
        _reload_dotenv_if_changed()

# =============================================
# ROUTES - Các trang chính
# =============================================

@app.route('/')
def home():
    """Trang chủ - Landing cho AI tạo thơ chúc Tết"""
    projects = session.get('projects')
    if not isinstance(projects, list):
        projects = []
    return render_template('home.html', projects=projects)


@app.route('/main')
@app.route('/main/')
def main():
    """Trang tạo thơ chúc Tết"""
    active = session.get('active_project')
    if isinstance(active, dict) and active.get('kind') == 'poem':
        return render_template('main.html', active_project=active)
    return render_template('main.html', active_project=None)


@app.route('/image')
@app.route('/image/')
def image():
    """Trang tạo ảnh chúc Tết từ prompt"""
    active = session.get('active_project')
    if isinstance(active, dict) and active.get('kind') == 'image':
        return render_template('image.html', active_project=active)
    return render_template('image.html', active_project=None)


@app.route('/music')
@app.route('/music/')
def music():
    """Trang tạo nhạc/audio chúc Tết từ prompt"""
    active = session.get('active_project')
    if isinstance(active, dict) and active.get('kind') == 'music':
        return render_template('music.html', active_project=active)
    return render_template('music.html', active_project=None)


def _cap_text(value: str | None, max_len: int) -> str:
    s = (value or '').strip()
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 1)].rstrip() + '…'


def _get_projects() -> list[dict]:
    projects = session.get('projects')
    if not isinstance(projects, list):
        return []
    return [p for p in projects if isinstance(p, dict)]


def _add_project(kind: str, title: str, data: dict):
    # Flask's default session is cookie-based (~4KB). Keep projects small.
    proj = {
        'id': uuid.uuid4().hex,
        'kind': kind,
        'title': _cap_text(title, 72) or 'Dự án',
        'data': data if isinstance(data, dict) else {},
        'ts': int(time.time()),
    }
    projects = _get_projects()
    projects.insert(0, proj)
    session['projects'] = projects[:8]
    session.modified = True


@app.route('/project/open/<int:idx>')
def open_project(idx: int):
    projects = _get_projects()
    if idx < 0 or idx >= len(projects):
        return redirect('/')
    session['active_project'] = projects[idx]
    session.modified = True

    kind = (projects[idx].get('kind') or '').strip().lower()
    if kind == 'image':
        return redirect('/image')
    if kind == 'music':
        return redirect('/music')
    return redirect('/main')


def _save_data_url_image_to_static(data_url: str) -> str | None:
    """Persist a data URL image into /static/generated and return its public URL."""
    if not isinstance(data_url, str):
        return None
    if not data_url.startswith('data:image/'):
        return None

    # data:image/png;base64,AAAA...
    m = re.match(r'^data:image/(?P<ext>[a-zA-Z0-9.+-]+);base64,(?P<b64>.+)$', data_url)
    if not m:
        return None

    ext = (m.group('ext') or 'png').lower()
    # Normalize a few common variants
    if ext in {'jpeg', 'jpg'}:
        ext = 'jpg'
    elif ext in {'png'}:
        ext = 'png'
    elif ext in {'webp'}:
        ext = 'webp'
    else:
        # Unknown/rare image types: store as png fallback name, but keep bytes.
        ext = ext.replace('.', '_')[:10] or 'png'

    b64 = m.group('b64') or ''
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return None

    out_dir = Path(app.root_path) / 'static' / 'generated'
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"img_{int(time.time())}_{uuid.uuid4().hex[:12]}.{ext}"
    out_path = out_dir / filename
    try:
        out_path.write_bytes(raw)
    except Exception:
        return None

    return f"/static/generated/{filename}"


def _delete_generated_static_file(image_url: str | None) -> None:
    if not image_url or not isinstance(image_url, str):
        return
    if not image_url.startswith('/static/generated/'):
        return
    filename = image_url.split('/static/generated/', 1)[-1].strip('/\\')
    if not filename or '..' in filename or '/' in filename or '\\' in filename:
        return
    path = Path(app.root_path) / 'static' / 'generated' / filename
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        return


@app.route('/project/delete/<int:idx>', methods=['POST'])
def delete_project(idx: int):
    projects = _get_projects()
    if idx < 0 or idx >= len(projects):
        return jsonify({'status': 'error', 'error': 'Invalid project index'}), 400

    removed = projects.pop(idx)
    session['projects'] = projects

    # If the active project equals the removed one, clear it.
    active = session.get('active_project')
    if isinstance(active, dict) and active == removed:
        session.pop('active_project', None)

    # Best-effort cleanup for persisted images.
    try:
        if isinstance(removed, dict) and (removed.get('kind') == 'image'):
            data = removed.get('data') if isinstance(removed.get('data'), dict) else {}
            image_url = data.get('image_url') if isinstance(data, dict) else None
            _delete_generated_static_file(image_url)
    except Exception:
        pass

    session.modified = True
    return jsonify({'status': 'success'})


@app.route('/api/suno-callback', methods=['POST'])
def api_suno_callback():
    """Webhook receiver for Kie.ai/Suno callbacks.

    Note: In local dev, Kie.ai cannot reach http://127.0.0.1.
    Use a public URL (ngrok/cloudflared) when you actually want callbacks.
    """
    _ = request.get_json(silent=True) or {}
    return jsonify({'status': 'received'})


def _get_openai_client():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None, 'Missing OPENAI_API_KEY environment variable'

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return None, f'OpenAI SDK not installed: {e}'

    return OpenAI(api_key=api_key), None


def _generate_image_with_openai(prompt: str):
    client, err = _get_openai_client()
    if err:
        return None, err

    model = (os.environ.get('OPENAI_IMAGE_MODEL') or '').strip() or 'gpt-image-1'
    size = (os.environ.get('OPENAI_IMAGE_SIZE') or '').strip() or '1024x1024'

    # Prefer base64 so frontend can display without remote URL.
    try:
        resp = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            response_format='b64_json',
        )
        data0 = (resp.data[0] if getattr(resp, 'data', None) else None)
        b64 = getattr(data0, 'b64_json', None) if data0 else None
        if b64:
            return {
                'image': f'data:image/png;base64,{b64}',
                'provider': 'openai',
                'model': model,
                'size': size,
            }, None
    except Exception as e:
        msg = str(e)
        # Common failure: org must be verified for image generation on some accounts.
        if 'must be verified' in msg.lower() or 'organization' in msg.lower():
            return None, f'OpenAI image request blocked by account/org settings: {e}'
        # Fall through to URL attempt.

    # Fallback: ask for URL if base64 is not supported for this model/account.
    try:
        resp = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
        )
        data0 = (resp.data[0] if getattr(resp, 'data', None) else None)
        url = getattr(data0, 'url', None) if data0 else None
        if url:
            return {
                'image_url': url,
                'provider': 'openai',
                'model': model,
                'size': size,
            }, None
        return None, 'OpenAI did not return image data'
    except Exception as e:
        return None, f'OpenAI image request failed: {e}'


def _default_music_plan():
    return {
        'bpm': 112,
        'key': 'A',
        'scale': 'minor',
        'progression': [1, 6, 3, 7],
        'swing': 0.06,
        'bars': 4,
        'melody_density': 0.55,
        'style': 'tet_pop',
        'seed': int(time.time()) % 1_000_000,
    }


def _coerce_music_plan(raw: dict | None):
    plan = _default_music_plan()
    if not isinstance(raw, dict):
        return plan

    def _clamp_int(v, lo, hi, default):
        try:
            n = int(v)
            return max(lo, min(hi, n))
        except Exception:
            return default

    def _clamp_float(v, lo, hi, default):
        try:
            n = float(v)
            return max(lo, min(hi, n))
        except Exception:
            return default

    bpm = _clamp_int(raw.get('bpm'), 70, 160, plan['bpm'])
    key = str(raw.get('key') or plan['key']).strip().upper()
    if key not in {'C','C#','DB','D','D#','EB','E','F','F#','GB','G','G#','AB','A','A#','BB','B'}:
        key = plan['key']

    scale = str(raw.get('scale') or plan['scale']).strip().lower()
    if scale not in {'major', 'minor', 'pentatonic'}:
        scale = plan['scale']

    prog = raw.get('progression')
    if isinstance(prog, list):
        degrees = []
        for x in prog[:8]:
            try:
                d = int(x)
            except Exception:
                continue
            if 1 <= d <= 7:
                degrees.append(d)
        if len(degrees) >= 2:
            plan['progression'] = degrees

    plan['bpm'] = bpm
    plan['key'] = key
    plan['scale'] = scale
    plan['swing'] = _clamp_float(raw.get('swing'), 0.0, 0.22, plan['swing'])
    plan['bars'] = _clamp_int(raw.get('bars'), 2, 8, plan['bars'])
    plan['melody_density'] = _clamp_float(raw.get('melody_density'), 0.15, 0.95, plan['melody_density'])
    plan['style'] = str(raw.get('style') or plan['style']).strip().lower()[:32] or plan['style']
    plan['seed'] = _clamp_int(raw.get('seed'), 0, 9_999_999, plan['seed'])

    return plan


def _generate_music_plan_with_openai(prompt: str):
    """Generate a compact 'music plan' (BPM/key/progression/pattern hints) for client-side WebAudio synthesis.

    This avoids saving audio on the server and enables real beat/melody/bass.
    """
    client, err = _get_openai_client()
    if err:
        return None, err

    text_model = os.environ.get('OPENAI_TEXT_MODEL', 'gpt-4o-mini')
    system = (
        'You are a music director for short upbeat Vietnamese New Year (Tết) instrumentals. '
        'Return ONLY valid JSON. No markdown. No explanations.\n'
        'Schema:\n'
        '{"bpm": int 70-160, "key": "C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B", '
        '"scale": "major|minor|pentatonic", "progression": [int degree 1-7], '
        '"swing": float 0-0.22, "bars": int 2-8, "melody_density": float 0.15-0.95, '
        '"style": "tet_pop|tet_hiphop|tet_house|lofi|traditional", "seed": int }\n'
        'Make it catchy with strong kick + bass + simple melody. '
        'If user asks for a vibe (happy/modern/traditional), reflect in bpm/scale/style.'
    )

    # Ask model for JSON plan; parse, validate, and coerce.
    try:
        text = None
        try:
            resp = client.responses.create(
                model=text_model,
                input=[
                    {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                    {'role': 'user', 'content': [{'type': 'text', 'text': (prompt or '').strip()}]},
                ],
                temperature=0.5,
            )
            text = getattr(resp, 'output_text', None)
        except Exception:
            text = None

        if not text:
            chat = client.chat.completions.create(
                model=text_model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': (prompt or '').strip()},
                ],
                temperature=0.5,
            )
            text = (chat.choices[0].message.content or '').strip()

        text = (text or '').strip()
        if not text:
            return None, 'OpenAI music plan generation returned empty result'

        raw = None
        try:
            raw = json.loads(text)
        except Exception:
            raw = None

        plan = _coerce_music_plan(raw if isinstance(raw, dict) else None)
        return {
            'kind': 'webaudio_plan',
            'provider': 'openai',
            'model': text_model,
            'plan': plan,
        }, None
    except Exception as e:
        return None, f'OpenAI music plan generation failed: {e}'


def _should_translate_to_english(text: str) -> bool:
    # Heuristic: if any non-ASCII character exists, assume non-English.
    return any(ord(ch) > 127 for ch in (text or ''))


def _translate_to_english_with_openai(text: str):
    """Translate user prompt to English for image generation providers that work better with English."""
    client, err = _get_openai_client()
    if err:
        return None, err

    model = os.environ.get('OPENAI_TRANSLATE_MODEL', os.environ.get('OPENAI_TEXT_MODEL', 'gpt-4o-mini'))
    system = (
        'You translate Vietnamese prompts into natural, concise English prompts for image generation. '
        'Rules: return ONLY the translated prompt text; keep style keywords; do not add quotes; '
        'do not add explanations; keep it one paragraph.'
    )

    try:
        translated = None
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                    {'role': 'user', 'content': [{'type': 'text', 'text': (text or '').strip()}]},
                ],
                temperature=0.2,
            )
            translated = getattr(resp, 'output_text', None)
        except Exception:
            translated = None

        if not translated:
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': (text or '').strip()},
                ],
                temperature=0.2,
            )
            translated = (chat.choices[0].message.content or '').strip()

        translated = (translated or '').strip()
        if not translated:
            return None, 'OpenAI translation returned empty result'
        return translated, None
    except Exception as e:
        return None, f'OpenAI translation failed: {e}'


def _get_gemini_api_key():
    # Support common env var names
    return (
        os.environ.get('GEMINI_API_KEY')
        or os.environ.get('GOOGLE_API_KEY')
        or os.environ.get('GOOGLE_GEMINI_API_KEY')
    )


def _get_stability_api_key():
    return (
        os.environ.get('STABILITY_API_KEY')
        or os.environ.get('STABILITY_KEY')
    )


def _get_suno_api_key():
    return (
        os.environ.get('SUNO_API_KEY')
        or os.environ.get('SUNO_KEY')
        or os.environ.get('KIE_API_KEY')
        or os.environ.get('KIE_KEY')
        or os.environ.get('MUSIC_API_KEY')
    )


def _get_suno_base_url():
    return (
        os.environ.get('SUNO_BASE_URL')
        or os.environ.get('SUNO_API_BASE_URL')
        or os.environ.get('KIE_BASE_URL')
        or os.environ.get('KIE_API_BASE_URL')
        or os.environ.get('MUSIC_BASE_URL')
        or ''
    ).strip().rstrip('/')


LYRICS_WORD_LIMIT = 300


def _trim_to_max_words(text: str, max_words: int = LYRICS_WORD_LIMIT) -> str:
    if not text:
        return ''

    count = 0
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    out: list[str] = []

    for ln in lines:
        raw = (ln or '').strip()
        if not raw:
            if out and out[-1] != '':
                out.append('')
            continue

        words = re.findall(r'\S+', raw)
        if not words:
            continue

        if count + len(words) <= max_words:
            out.append(raw)
            count += len(words)
            continue

        remaining = max_words - count
        if remaining <= 0:
            break

        out.append(' '.join(words[:remaining]))
        count = max_words
        break

    while out and out[0] == '':
        out.pop(0)
    while out and out[-1] == '':
        out.pop()

    return ('\n'.join(out)).strip()


def _generate_lyrics_with_openai(prompt: str, length: str | None = None, note: str | None = None):
    """Generate/Remake Vietnamese lyrics for Suno.

    This is text-only (no TTS). Keep it short so Suno can render quickly.
    """
    client, err = _get_openai_client()
    if err:
        return None, err

    text_model = os.environ.get('OPENAI_TEXT_MODEL', 'gpt-4o-mini')
    # Keep lyrics short enough for fast music generation and predictable UX.
    # "200 chữ" is interpreted as ~200 whitespace-delimited words.
    max_words = 200

    length_key = (length or 'same').strip().lower()
    if length_key not in ('shorter', 'same', 'longer'):
        length_key = 'same'

    length_rule = {
        'shorter': '6–8 dòng',
        'same': '8–12 dòng',
        'longer': '14–20 dòng',
    }[length_key]

    user_note = (note or '').strip()

    system = (
        'Bạn là nhạc sĩ/thi sĩ tiếng Việt. Hãy viết (hoặc viết lại) LỜI BÀI HÁT chúc Tết để AI tạo nhạc. '
        f'Yêu cầu: {length_rule}; có nhịp; câu ngắn; dễ hát; tránh ký tự lạ; hạn chế emoji; không tiêu đề; không giải thích. '
        f'Tối đa {max_words} từ. '
        'Nếu người dùng nêu người nhận/tông giọng/từ khóa thì lồng ghép khéo léo. '
        + (f'Chú thích của người dùng (ưu tiên làm theo): {user_note}' if user_note else '')
    )

    user_prompt = (prompt or '').strip()
    if user_note:
        user_prompt = f"Đề bài / lời gốc:\n{user_prompt}\n\nChú thích:\n{user_note}".strip()

    try:
        lyrics = None
        try:
            resp = client.responses.create(
                model=text_model,
                input=[
                    {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                    {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt}]},
                ],
                temperature=0.8,
            )
            lyrics = getattr(resp, 'output_text', None)
        except Exception:
            lyrics = None

        if not lyrics:
            chat = client.chat.completions.create(
                model=text_model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=0.8,
            )
            lyrics = (chat.choices[0].message.content or '').strip()

        lyrics = (lyrics or '').strip()
        lyrics = _trim_to_max_words(lyrics, max_words)
        if not lyrics:
            return None, 'OpenAI lyrics generation returned empty result'
        return {
            'lyrics': lyrics,
            'provider': 'openai',
            'model': text_model,
        }, None
    except Exception as e:
        return None, f'OpenAI lyrics generation failed: {e}'


def _call_suno_generate(prompt: str, lyrics: str | None = None, extra_fields: dict | None = None):
    """Call a Suno API provider (3rd-party) to generate a track.

    IMPORTANT: Suno API providers differ in endpoint paths, auth, and response schema.
    Configure via env vars:
      - SUNO_BASE_URL (required)
      - SUNO_API_KEY (optional depending on provider)
      - SUNO_GENERATE_PATH (required) e.g. /generate
      - SUNO_STATUS_PATH (optional) e.g. /status/{id}
    The code expects the generate response to return either:
      - {"audio_url": "https://...mp3"}
      - or {"id": "..."} and then status endpoint returns {"audio_url": "..."}
    """
    base = _get_suno_base_url()
    if not base:
        return None, 'Missing SUNO_BASE_URL (or SUNO_API_BASE_URL)'

    generate_path = (os.environ.get('SUNO_GENERATE_PATH') or '').strip()
    if not generate_path:
        return None, 'Missing SUNO_GENERATE_PATH (provider-specific)'

    status_path = (os.environ.get('SUNO_STATUS_PATH') or '').strip()
    api_key = _get_suno_api_key()

    try:
        import requests  # type: ignore
    except Exception as e:
        return None, f'Requests not installed: {e}'

    url = f'{base}{generate_path if generate_path.startswith("/") else "/" + generate_path}'

    prompt_field = (os.environ.get('SUNO_PROMPT_FIELD') or 'prompt').strip() or 'prompt'
    lyrics_field = (os.environ.get('SUNO_LYRICS_FIELD') or 'lyrics').strip() or 'lyrics'

    payload: dict = {
        prompt_field: (prompt or '').strip(),
    }
    if lyrics:
        payload[lyrics_field] = lyrics

    if isinstance(extra_fields, dict) and extra_fields:
        # Only merge JSON-serializable primitives/structures
        try:
            json.dumps(extra_fields)
            payload.update(extra_fields)
        except Exception:
            pass

    def _env_bool(name: str, default: bool | None = None) -> bool | None:
        raw = os.environ.get(name)
        if raw is None:
            return default
        s = str(raw).strip().lower()
        if s in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if s in {'0', 'false', 'no', 'n', 'off'}:
            return False
        return default

    # Kie.ai Suno API requirements: model + callBackUrl are required, tasks are async (taskId).
    is_kie = 'kie.ai' in base.lower()
    if is_kie:
        # Allow request to pass either snake_case or camelCase.
        if 'custom_mode' in payload and 'customMode' not in payload:
            payload['customMode'] = bool(payload.get('custom_mode'))
        if 'instrumental' in payload:
            payload['instrumental'] = bool(payload.get('instrumental'))

        payload.setdefault('model', (os.environ.get('SUNO_MODEL') or os.environ.get('KIE_MODEL') or 'V4_5PLUS').strip())
        payload.setdefault('customMode', bool(_env_bool('SUNO_CUSTOM_MODE', False)))
        payload.setdefault('instrumental', bool(_env_bool('SUNO_INSTRUMENTAL', True)))

        callback_url = (
            os.environ.get('SUNO_CALLBACK_URL')
            or os.environ.get('KIE_CALLBACK_URL')
            or ''
        ).strip()
        if not callback_url:
            return None, 'Missing SUNO_CALLBACK_URL (required by Kie.ai Suno API)'
        payload.setdefault('callBackUrl', callback_url)

    extra_json = (os.environ.get('SUNO_EXTRA_JSON') or '').strip()
    if extra_json:
        try:
            extra_obj = json.loads(extra_json)
            if isinstance(extra_obj, dict):
                payload.update(extra_obj)
        except Exception:
            # Ignore malformed extra json
            pass

    headers = {
        'Content-Type': 'application/json',
    }
    # Try common auth patterns; providers vary.
    if api_key:
        headers.setdefault('Authorization', f'Bearer {api_key}')
        headers.setdefault('X-API-Key', api_key)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
    except Exception as e:
        return None, f'Network error calling Suno provider: {e}'

    if r.status_code >= 400:
        body = (r.text or '').strip()
        snippet = body[:800] if body else '(empty body)'
        return None, f'Suno generate failed: {r.status_code} - {snippet}'

    # Parse JSON (some providers return list at top-level)
    try:
        data = r.json() if r.content else {}
    except Exception:
        body = (r.text or '').strip()
        snippet = body[:800] if body else '(empty body)'
        return None, f'Suno generate did not return valid JSON: {snippet}'

    def _as_str(x):
        return x.strip() if isinstance(x, str) and x.strip() else None

    def _find_first(obj, wanted_keys: set[str]):
        """Depth-first search for first matching key in dict/list structures."""
        if isinstance(obj, dict):
            for k in wanted_keys:
                if k in obj:
                    v = obj.get(k)
                    if v is not None:
                        return v
            for v in obj.values():
                found = _find_first(v, wanted_keys)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _find_first(item, wanted_keys)
                if found is not None:
                    return found
        return None

    def _find_audio_url(obj):
        # Prefer explicit audio keys first
        v = _find_first(obj, {
            'audio_url', 'audioUrl', 'audio', 'audioSrc', 'audio_src', 'mp3_url', 'mp3Url',
            'track_url', 'trackUrl', 'song_url', 'songUrl',
        })
        s = _as_str(v)
        if s:
            return s

        # Fallback: look for any URL-ish string in common containers
        v2 = _find_first(obj, {'url', 'file', 'file_url', 'fileUrl'})
        s2 = _as_str(v2)
        if s2 and any(ext in s2.lower() for ext in ('.mp3', '.wav', '.m4a', '.aac', '.ogg')):
            return s2
        return None

    def _find_job_id(obj):
        v = _find_first(obj, {
            'id', 'job_id', 'jobId', 'task_id', 'taskId', 'record_id', 'recordId',
            'song_id', 'songId', 'uuid'
        })
        return _as_str(v)

    # Surface provider-side errors even if HTTP 200
    if isinstance(data, dict):
        code = data.get('code')
        status = data.get('status')
        okish = str(status).lower() in {'success', 'ok', 'succeeded'}
        if (isinstance(code, int) and code not in (0, 200)) or (status is not None and not okish):
            msg = data.get('message') or data.get('msg') or data.get('error')
            if msg:
                return None, f'Suno provider error: {msg}'

        if data.get('error') and not data.get('audio_url'):
            return None, f"Suno provider error: {data.get('error')}"

    audio_url = _find_audio_url(data)
    if audio_url:
        return {
            'audio_url': audio_url,
            'provider': 'suno',
        }, None

    job_id = _find_job_id(data)
    if not job_id:
        # Provide a helpful hint (keys + snippet) for mapping this provider
        body = (r.text or '').strip()
        snippet = body[:800] if body else '(empty body)'
        keys = list(data.keys())[:40] if isinstance(data, dict) else []
        return None, f'Suno provider did not return audio_url or job id. keys={keys} body={snippet}'

    if not status_path:
        return None, 'Suno provider returned a job id but SUNO_STATUS_PATH is not configured'

    # Poll status
    # Kie.ai tasks can take longer than generic Suno providers.
    max_wait_raw = (os.environ.get('SUNO_MAX_WAIT_SECONDS') or '').strip()
    if max_wait_raw:
        max_wait_s = int(max_wait_raw)
    else:
        max_wait_s = 420 if is_kie else 180

    poll_every_raw = (os.environ.get('SUNO_POLL_SECONDS') or '').strip()
    if poll_every_raw:
        poll_every_s = float(poll_every_raw)
    else:
        poll_every_s = 3.0 if is_kie else 2.0
    deadline = time.time() + max(10, min(max_wait_s, 600))

    status_url = status_path.replace('{id}', str(job_id).strip())
    status_url = f'{base}{status_url if status_url.startswith("/") else "/" + status_url}'

    while time.time() < deadline:
        try:
            rs = requests.get(status_url, headers=headers, timeout=60)
        except Exception as e:
            return None, f'Network error polling Suno status: {e}'

        if rs.status_code >= 400:
            body = (rs.text or '').strip()
            snippet = body[:800] if body else '(empty body)'
            return None, f'Suno status failed: {rs.status_code} - {snippet}'

        try:
            sd = rs.json() if rs.content else {}
        except Exception:
            body = (rs.text or '').strip()
            snippet = body[:800] if body else '(empty body)'
            return None, f'Suno status did not return valid JSON: {snippet}'

        audio_url = _find_audio_url(sd)
        if audio_url:
            return {
                'audio_url': audio_url,
                'provider': 'suno',
                'job_id': str(job_id).strip(),
            }, None

        if isinstance(sd, dict):
            state = (sd.get('status') or sd.get('state') or '').lower()
            if state in {'failed', 'error'}:
                return None, f'Suno job failed: {sd}'

        time.sleep(poll_every_s)

    return None, (
        f'Suno job timed out waiting for audio_url (job_id={str(job_id).strip()}). '
        'Try increasing SUNO_MAX_WAIT_SECONDS (max 600) or check provider status endpoint.'
    )


def _openai_size_to_aspect_ratio(size: str) -> str:
    s = (size or '').strip().lower()
    mapping = {
        '1024x1024': '1:1',
        '512x512': '1:1',
        '256x256': '1:1',
        '1024x768': '4:3',
        '768x1024': '3:4',
        '1024x576': '16:9',
        '576x1024': '9:16',
    }
    return mapping.get(s, '1:1')


_GEMINI_MODELS_CACHE = {
    # key: (api_key, version) -> {'ts': float, 'models': list[dict]}
}


def _gemini_list_models(api_key: str, version: str, cache_ttl_seconds: int = 300):
    """Return list of models visible to this API key for a given API version."""
    cache_key = (api_key, version)
    now = time.time()
    cached = _GEMINI_MODELS_CACHE.get(cache_key)
    if cached and (now - float(cached.get('ts', 0))) < cache_ttl_seconds:
        return cached.get('models') or [], None

    try:
        import requests  # type: ignore
    except Exception as e:
        return [], f'Requests not installed: {e}'

    url = f'https://generativelanguage.googleapis.com/{version}/models?key={api_key}'
    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        return [], f'Network error calling Gemini ListModels: {e}'

    if r.status_code >= 400:
        body = (r.text or '').strip()
        snippet = body[:700] if body else '(empty body)'
        return [], f'Gemini ListModels failed ({version}): {r.status_code} - {snippet}'

    data = r.json() if r.content else {}
    models = data.get('models') or []
    if not isinstance(models, list):
        models = []

    _GEMINI_MODELS_CACHE[cache_key] = {'ts': now, 'models': models}
    return models, None


def _gemini_models_supporting_generate_content(models: list):
    supported = []
    for m in models or []:
        if not isinstance(m, dict):
            continue
        methods = m.get('supportedGenerationMethods') or m.get('supported_generation_methods') or []
        if isinstance(methods, list) and any(str(x).endswith('generateContent') for x in methods):
            supported.append(m)
    return supported


def _pick_gemini_image_model(models: list, preferred: str | None = None):
    """Pick the best available image-capable model from ListModels."""
    names = []
    for m in models or []:
        if isinstance(m, dict) and isinstance(m.get('name'), str):
            # name can be like 'models/gemini-2.5-flash'
            names.append(m['name'].replace('models/', ''))

    preferred = (preferred or '').strip()
    if preferred:
        # If preferred is present in list, keep it.
        if preferred in names:
            return preferred
        # Also accept if it was returned as models/<name>
        if f'models/{preferred}' in [m.get('name') for m in models if isinstance(m, dict)]:
            return preferred

    # Heuristic: prefer anything that looks like an image model.
    # (Exact availability varies by key/region; ListModels is the source of truth.)
    candidates = []
    for n in names:
        ln = n.lower()
        score = 0
        # Prefer Imagen 3 latest variant if present.
        if 'imagen-3.0-generate-002' in ln:
            score += 200
        if 'flash-image' in ln:
            score += 100
        if 'image' in ln:
            score += 50
        if 'imagen' in ln:
            score += 40
        if '2.5' in ln:
            score += 10
        candidates.append((score, n))

    candidates.sort(reverse=True)
    if candidates and candidates[0][0] > 0:
        return candidates[0][1]

    # If nothing looks image-capable, return None (avoid calling text-only models).
    return None


def _generate_image_with_gemini(prompt: str):
    api_key = _get_gemini_api_key()
    if not api_key:
        return None, 'Missing GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable'

    # Keep compatibility with existing OPENAI_IMAGE_SIZE env by mapping to aspect ratio.
    openai_size = os.environ.get('OPENAI_IMAGE_SIZE', '1024x1024')
    aspect_ratio = os.environ.get('GEMINI_IMAGE_ASPECT_RATIO', _openai_size_to_aspect_ratio(openai_size))

    try:
        import requests  # type: ignore
    except Exception as e:
        return None, f'Requests not installed: {e}'

    # Gemini API uses generateContent with the schema:
    # contents -> [{ role, parts: [{ text }] }]
    # Some models can return images as inlineData parts.
    preferred_model = (os.environ.get('GEMINI_IMAGE_MODEL') or '').strip() or 'gemini-2.5-flash-image'
    endpoint_candidates = ['v1beta', 'v1']

    # Payload variants: keep the schema valid; optionally try responseModalities.
    base_contents = [
        {
            'role': 'user',
            'parts': [
                {
                    'text': prompt,
                }
            ],
        }
    ]

    # IMPORTANT: Do NOT send generationConfig.responseModalities.
    # Many text-only models (e.g., gemini-2.5-pro) reject it with 400.
    # Image models should return inlineData without requiring responseModalities.
    payload_candidates = [
        {
            'contents': base_contents,
        }
    ]

    last_error = None
    data = None
    chosen_model = None
    chosen_version = None

    # Discover models (per version) to avoid blind 404 calls.
    # If ListModels fails (e.g., permission), fall back to trying preferred_model directly.
    discovered_models_by_version = {}
    for version in endpoint_candidates:
        models, err = _gemini_list_models(api_key, version)
        if err:
            discovered_models_by_version[version] = {'models': [], 'err': err}
            continue
        supporting = _gemini_models_supporting_generate_content(models)
        discovered_models_by_version[version] = {'models': supporting, 'err': None}

    for version in endpoint_candidates:
        version_models = discovered_models_by_version.get(version, {}).get('models') or []
        picked = _pick_gemini_image_model(version_models, preferred=preferred_model)
        model_try_order = []
        if preferred_model:
            model_try_order.append(preferred_model)
        if picked and picked not in model_try_order:
            model_try_order.append(picked)

        # If ListModels worked, only try models that are actually visible.
        visible_names = {
            (m.get('name') or '').replace('models/', '')
            for m in version_models
            if isinstance(m, dict)
        }
        def _looks_like_image_model(name: str) -> bool:
            ln = (name or '').lower()
            return ('imagen' in ln) or ('-image' in ln) or ('image' in ln)

        if visible_names:
            # Only attempt models that look image-capable.
            visible_image_names = [n for n in visible_names if _looks_like_image_model(n)]
            if visible_image_names:
                model_try_order = [m for m in model_try_order if m in visible_image_names] or [visible_image_names[0]]
            else:
                model_try_order = []

        # If we have nothing to try, surface ListModels error.
        if not model_try_order:
            err = discovered_models_by_version.get(version, {}).get('err')
            last_error = (
                err
                or f'No image-capable Gemini/Imagen models available for version={version}. '
                'Your API key may not have access to Imagen/image models, or billing/quota is not enabled.'
            )
            continue

        for model in model_try_order:
            url = f'https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={api_key}'
            for payload in payload_candidates:
                try:
                    r = requests.post(
                        url,
                        headers={'Content-Type': 'application/json'},
                        json=payload,
                        timeout=90,
                    )
                except Exception as e:
                    last_error = f'Network error calling Gemini: {e}'
                    continue

                # 404: model or endpoint not available for this key/version.
                if r.status_code == 404:
                    last_error = (
                        f'Gemini image request failed (model={model} endpoint={version}/generateContent): 404. '
                        f'Model not visible for this API key/version; check ListModels or change GEMINI_IMAGE_MODEL.'
                    )
                    break

                # 429: quota/billing/rate-limit
                if r.status_code == 429:
                    body = (r.text or '').strip()
                    snippet = body[:700] if body else '(empty body)'
                    last_error = (
                        f'Gemini image request failed (model={model} endpoint={version}/generateContent): 429 RESOURCE_EXHAUSTED - {snippet}. '
                        'This typically means your project has zero quota for this model (free-tier limit 0) or billing/quota is not enabled.'
                    )
                    continue

                if r.status_code >= 400:
                    body = (r.text or '').strip()
                    snippet = body[:700] if body else '(empty body)'
                    last_error = (
                        f'Gemini image request failed (model={model} endpoint={version}/generateContent): '
                        f'{r.status_code} - {snippet}'
                    )
                    continue

                chosen_model = model
                chosen_version = version
                data = r.json() if r.content else {}
                break

            if data is not None:
                break
        if data is not None:
            break

    if data is None:
        return None, last_error or 'Gemini image request failed (no successful response)'

    # Gemini image-capable models return image bytes in inlineData.
    try:
        candidates = data.get('candidates') or []
        parts = (((candidates[0] or {}).get('content') or {}).get('parts') or []) if candidates else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get('inlineData') or part.get('inline_data')
            if isinstance(inline, dict):
                b64 = inline.get('data')
                mime = inline.get('mimeType') or inline.get('mime_type') or 'image/png'
                if b64:
                    return {
                        'image': f'data:{mime};base64,{b64}',
                        'model': chosen_model,
                        'aspect_ratio': aspect_ratio,
                        'provider': 'gemini',
                    }, None
    except Exception:
        pass

    # If we got here, the endpoint returned 2xx but we couldn't find image bytes.
    # Include a hint about which endpoint succeeded to ease debugging.
    hint = ''
    if chosen_model and chosen_version:
        hint = f' (2xx from {chosen_version} {chosen_model}, but no inline image bytes found)'
    return None, 'Gemini did not return image data' + hint


def _openai_size_to_width_height(size: str) -> tuple[int, int]:
    s = (size or '').strip().lower()
    try:
        if 'x' in s:
            w_str, h_str = s.split('x', 1)
            w = int(w_str)
            h = int(h_str)
            return max(256, min(w, 2048)), max(256, min(h, 2048))
    except Exception:
        pass
    return 1024, 1024


def _generate_image_with_stability(prompt: str):
    api_key = _get_stability_api_key()
    if not api_key:
        return None, 'Missing STABILITY_API_KEY environment variable'

    try:
        import requests  # type: ignore
    except Exception as e:
        return None, f'Requests not installed: {e}'

    # Keep compatibility with existing OPENAI_IMAGE_SIZE env.
    openai_size = os.environ.get('OPENAI_IMAGE_SIZE', '1024x1024')
    width, height = _openai_size_to_width_height(openai_size)
    aspect_ratio = os.environ.get('STABILITY_ASPECT_RATIO', _openai_size_to_aspect_ratio(openai_size))

    # Prefer Stability v2beta endpoints (return raw image bytes).
    # If unavailable for your account, fall back to v1 generation endpoint (returns base64 JSON artifacts).
    v2_candidates = [
        'https://api.stability.ai/v2beta/stable-image/generate/ultra',
        'https://api.stability.ai/v2beta/stable-image/generate/core',
    ]

    # Minimal, safe prompt pass-through.
    # Do not attempt to add new UX/features; backend only.
    for url in v2_candidates:
        try:
            r = requests.post(
                url,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Accept': 'image/*',
                },
                data={
                    'prompt': prompt,
                    'output_format': 'png',
                    'aspect_ratio': aspect_ratio,
                },
                timeout=90,
            )
        except Exception as e:
            last = f'Network error calling Stability: {e}'
            continue

        if r.status_code == 401 or r.status_code == 403:
            body = (r.text or '').strip()
            snippet = body[:700] if body else '(empty body)'
            return None, f'Stability auth failed: {r.status_code} - {snippet}'

        if r.status_code == 429:
            body = (r.text or '').strip()
            snippet = body[:700] if body else '(empty body)'
            return None, f'Stability rate-limit/quota hit: 429 - {snippet}'

        if r.status_code == 404:
            # Endpoint not available for this account/API version.
            last = f'Stability endpoint not found: {url} (404)'
            continue

        if r.status_code >= 400:
            body = (r.text or '').strip()
            snippet = body[:700] if body else '(empty body)'
            last = f'Stability request failed: {r.status_code} - {snippet}'
            continue

        content_type = (r.headers.get('Content-Type') or '').split(';', 1)[0].strip() or 'image/png'
        b64 = base64.b64encode(r.content).decode('ascii') if r.content else ''
        if not b64:
            last = 'Stability returned 2xx but empty image body'
            continue

        return {
            'image': f'data:{content_type};base64,{b64}',
            'provider': 'stability',
            'engine': os.environ.get('STABILITY_ENGINE', ''),
            'aspect_ratio': aspect_ratio,
        }, None

    # Fallback: Stability v1 generation endpoint.
    engine = (os.environ.get('STABILITY_ENGINE') or '').strip() or 'stable-diffusion-xl-1024-v1-0'
    v1_url = f'https://api.stability.ai/v1/generation/{engine}/text-to-image'
    body = {
        'text_prompts': [{'text': prompt}],
        'cfg_scale': float(os.environ.get('STABILITY_CFG_SCALE', '7')),
        'height': height,
        'width': width,
        'samples': 1,
        'steps': int(os.environ.get('STABILITY_STEPS', '30')),
    }

    try:
        r = requests.post(
            v1_url,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            json=body,
            timeout=90,
        )
    except Exception as e:
        return None, f'Network error calling Stability v1: {e}'

    if r.status_code >= 400:
        txt = (r.text or '').strip()
        snippet = txt[:700] if txt else '(empty body)'
        return None, f'Stability v1 request failed ({engine}): {r.status_code} - {snippet}'

    data = r.json() if r.content else {}
    artifacts = data.get('artifacts') or []
    if isinstance(artifacts, list) and artifacts:
        art0 = artifacts[0] if isinstance(artifacts[0], dict) else None
        if art0 and art0.get('base64'):
            return {
                'image': f"data:image/png;base64,{art0['base64']}",
                'provider': 'stability',
                'engine': engine,
                'aspect_ratio': aspect_ratio,
            }, None

    return None, 'Stability did not return image data'


def _looks_off_topic(prompt: str) -> bool:
    """Heuristic filter to keep /api/generate-poem focused on Tet poems."""
    p = (prompt or '').strip().lower()
    if not p:
        return True

    # Strong signals for general Q&A / troubleshooting / coding help
    off_topic_markers = [
        'là gì', 'là sao', 'nghĩa là', 'giải thích', 'tại sao', 'vì sao',
        'how to', 'what is', 'why', 'explain',
        'code', 'lỗi', 'bug', 'traceback', 'stack', 'python', 'flask', 'javascript',
        'sql', 'api', 'server', 'deploy', 'cài đặt', 'setup', 'hướng dẫn',
    ]
    if any(m in p for m in off_topic_markers):
        return True

    # If it's mostly a question, it's likely off-topic.
    if ('?' in p) and not any(k in p for k in ['thơ', 'chúc', 'tết', 'xuân', 'năm mới']):
        return True

    return False


@app.route('/api/generate-poem', methods=['POST'])
def api_generate_poem():
    """API: Generate a Tet wish poem from a single prompt."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get('prompt') or '').strip()
    poem_style = (data.get('poem_style') or '').strip()
    if not prompt:
        return jsonify({'status': 'error', 'error': 'Missing prompt'}), 400

    allowed_styles = {
        'tu_do': 'Thơ tự do (có vần điệu)',
        'luc_bat': 'Lục bát',
        'song_that_luc_bat': 'Song thất lục bát',
        'that_ngon_tu_tuyet': 'Thất ngôn tứ tuyệt (4 câu, 7 chữ)',
        'that_ngon_bat_cu': 'Thất ngôn bát cú (8 câu, 7 chữ)',
        'cau_doi': 'Câu đối (2 vế)',
    }
    if poem_style not in allowed_styles:
        poem_style = 'tu_do'

    trimmed_prompt = _trim_to_max_words(prompt)
    prompt_truncated = trimmed_prompt != prompt
    prompt = trimmed_prompt

    if _looks_off_topic(prompt):
        return (
            jsonify(
                {
                    'status': 'error',
                    'error': 'Chỉ hỗ trợ viết thơ chúc Tết. Hãy gửi gợi ý như: người nhận, không khí, lời chúc, vài từ khóa.',
                }
            ),
            400,
        )

    client, err = _get_openai_client()
    if err:
        return jsonify({'status': 'error', 'error': err}), 500

    model = (
        os.environ.get('OPENAI_POEM_MODEL')
        or os.environ.get('OPENAI_TEXT_MODEL')
        or 'gpt-4.1-mini'
    )

    style_instructions = {
        'tu_do': (
            'Thể thơ: THƠ TỰ DO (truyền thống, dễ đọc). Nếu prompt nói “ngắn” thì ưu tiên 4 dòng; nếu không thì 4–8 dòng. '
            'Câu gọn (ưu tiên 6–10 tiếng mỗi dòng) để nhịp đều. '
            'Bắt buộc có VẦN CHÂN rõ ràng: chọn 1 VẦN chủ đạo (ví dụ: “-ang”, “-ơi”, “-a”) và gieo vần chân XUYÊN SUỐT bài (phần lớn các dòng; nếu 4 dòng thì tối thiểu 3 dòng cùng vần). '
            'Không đặt dấu câu sau tiếng gieo vần. '
            'Tránh kết thúc nhiều dòng bằng cùng một TỪ y hệt (có thể đổi từ nhưng giữ cùng vần) để thơ “đã tai” mà không lặp.'
        ),
        'luc_bat': (
            'Thể thơ: LỤC BÁT. Viết 4–8 dòng (số dòng CHẴN) theo cặp 6/8 (luân phiên). '
            'Bắt buộc giữ đúng số tiếng mỗi dòng (đếm theo các cụm tách bằng dấu cách): 6 tiếng, rồi 8 tiếng. '
            'Quy tắc trình bày: mỗi tiếng cách nhau đúng 1 dấu cách; không dính chữ, không viết tắt. '
            'Gieo vần theo lục bát: tiếng thứ 6 của câu 6 vần với tiếng thứ 6 của câu 8; '
            'tiếng cuối câu 8 vần với tiếng thứ 6 câu 6 tiếp theo. '
            'Ưu tiên chọn 1 vần chủ đạo cho TIẾNG CUỐI các câu 8 (câu 2,4,6,8) để bài “đã tai”, tránh đổi vần liên tục. '
            'Trước khi trả lời, hãy tự đếm số tiếng từng câu và sửa cho đúng; không hiển thị phần kiểm tra.'
        ),
        'song_that_luc_bat': (
            'Thể thơ: SONG THẤT LỤC BÁT. Viết 4 câu theo nhịp 7/7/6/8. '
            'Bắt buộc giữ số tiếng (đếm theo các cụm tách bằng dấu cách): 7, 7, 6, 8. '
            'Gieo vần mượt, ưu tiên vần chân: câu 1–2–4 cùng vần hoặc gần vần. '
            'Trước khi trả lời, hãy tự đếm số tiếng từng câu và sửa cho đúng; không hiển thị phần kiểm tra.'
        ),
        'that_ngon_tu_tuyet': (
            'Thể thơ: THẤT NGÔN TỨ TUYỆT. Viết đúng 4 câu, mỗi câu 7 tiếng (7 chữ). '
            'Bắt buộc có VẦN CHÂN: chọn 1 vần (ví dụ: “-ang”, “-ơi”, “-a”) và gieo vần chân ở cuối câu 1, 2, 4 (câu 3 có thể khác). '
            'Không đặt dấu câu sau tiếng gieo vần. '
            'Ngôn từ cổ phong vừa phải, có hình ảnh mùa xuân/Tết. '
            'Bắt buộc đếm đúng 7 tiếng mỗi câu (đếm theo các cụm tách bằng dấu cách). '
            'Trước khi trả lời, hãy tự đếm số tiếng từng câu và sửa cho đúng; không hiển thị phần kiểm tra.'
        ),
        'that_ngon_bat_cu': (
            'Thể thơ: THẤT NGÔN BÁT CÚ (Đường luật). Viết đúng 8 câu, mỗi câu 7 tiếng (7 chữ). '
            'Bắt buộc có VẦN CHÂN: chọn 1 vần (ví dụ: “-ang”, “-ơi”, “-a”) và gieo vần chân ở cuối câu 1, 2, 4, 6, 8. '
            'Không đặt dấu câu sau tiếng gieo vần. '
            'Cố gắng có đối ý/đối từ ở cặp câu 3–4 và 5–6 (không cần giải thích luật). '
            'Bắt buộc đếm đúng 7 tiếng mỗi câu (đếm theo các cụm tách bằng dấu cách). '
            'Trước khi trả lời, hãy tự đếm số tiếng từng câu và sửa cho đúng; không hiển thị phần kiểm tra.'
        ),
        'cau_doi': (
            'Thể loại: CÂU ĐỐI. Viết đúng 2 câu (2 vế), cân đối độ dài, đối ý/đối từ, giàu khí Tết. '
            'Ưu tiên vần ở cuối 2 vế (cùng vần hoặc gần vần).'
        ),
    }
    system = (
        'Bạn là một thi sĩ Việt Nam, chuyên viết thơ chúc Tết có nhạc tính “đã tai”. '
        'Mục tiêu bắt buộc: VẦN – NHỊP – LỰC thơ phải mạnh. Đọc lên phải trôi, êm, có điểm rơi ở cuối câu (vần chân) và có nhịp tự nhiên. '
        'Kỹ thuật (áp dụng tinh tế, không liệt kê ra ngoài): '
        '- Chọn 1 trường hình ảnh mùa xuân/Tết (mai/đào/lộc/sum vầy/đèn hoa) và phát triển nhất quán. '
        '- Dùng từ ngữ “đắt”, gợi hình, tránh sáo rỗng; không viết kiểu khẩu hiệu. '
        '- Ưu tiên vần chân rõ (đồng vần hoặc gần vần), tránh vần gượng; hạn chế lặp từ vô ý. '
        '- Tạo nhịp bằng cụm 2–3 tiếng, điểm dừng hợp lý; có thể dùng đối ý/đối từ ở cặp câu nếu phù hợp. '
        '- Tránh các kết hợp từ kì cục/khó đọc; tránh câu quá dài gây hụt nhịp. '
        'Nếu người dùng cung cấp người nhận/từ khóa, hãy lồng vào tự nhiên (không liệt kê). '
        f"{style_instructions.get(poem_style, style_instructions['tu_do'])} "
        'Tự kiểm tra trước khi trả lời: đọc thầm như đọc thơ Việt; sửa chỗ sượng/thiếu nhịp; đảm bảo có vần điệu rõ ràng theo thể thơ. '
        'Ràng buộc định dạng: chỉ trả về NỘI DUNG (không tiêu đề, không lời dẫn, không đánh số), mỗi câu một dòng.'
    )

    try:
        # Prefer Responses API; fallback to Chat Completions if needed
        poem_text = None
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {
                        'role': 'system',
                        'content': [{'type': 'text', 'text': system}],
                    },
                    {
                        'role': 'user',
                        'content': [{'type': 'text', 'text': prompt}],
                    },
                ],
                temperature=0.9,
            )
            # New SDKs provide output_text
            poem_text = getattr(resp, 'output_text', None)
        except Exception:
            poem_text = None

        if not poem_text:
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.9,
            )
            poem_text = (chat.choices[0].message.content or '').strip()

        def _split_lines(text: str) -> list[str]:
            lines = [(ln or '').strip() for ln in (text or '').splitlines()]
            return [ln for ln in lines if ln]

        def _count_syllables_by_space(line: str) -> int:
            # Approximate Vietnamese "tiếng" count: space-separated tokens.
            parts = [p for p in (line or '').strip().split(' ') if p]
            return len(parts)

        def _last_token_normalized(line: str) -> str:
            raw = (line or '').strip()
            if not raw:
                return ''
            last = raw.split()[-1]
            last = last.strip('"\'“”‘’.,;:!?…–—()[]{}')
            return last.lower()

        def _strip_vn_diacritics(s: str) -> str:
            src = (
                'àáạảãâầấậẩẫăằắặẳẵ'
                'èéẹẻẽêềếệểễ'
                'ìíịỉĩ'
                'òóọỏõôồốộổỗơờớợởỡ'
                'ùúụủũưừứựửữ'
                'ỳýỵỷỹ'
                'đ'
                'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ'
                'ÈÉẸẺẼÊỀẾỆỂỄ'
                'ÌÍỊỈĨ'
                'ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ'
                'ÙÚỤỦŨƯỪỨỰỬỮ'
                'ỲÝỴỶỸ'
                'Đ'
            )
            dst = (
                'aaaaaaaaaaaaaaaaa'
                'eeeeeeeeeee'
                'iiiii'
                'ooooooooooooooooo'
                'uuuuuuuuuuu'
                'yyyyy'
                'd'
                'AAAAAAAAAAAAAAAAA'
                'EEEEEEEEEEE'
                'IIIII'
                'OOOOOOOOOOOOOOOOO'
                'UUUUUUUUUUU'
                'YYYYY'
                'D'
            )
            table = {ord(src[i]): dst[i] for i in range(len(src))}
            return (s or '').translate(table)

        def _rhyme_key_from_token(token: str) -> str:
            t = _strip_vn_diacritics((token or '').strip().lower())
            t = t.strip('"\'“”‘’.,;:!?…–—()[]{}')
            if not t:
                return ''

            common = [
                'oang', 'uang', 'uynh', 'uyen', 'uong', 'ieng',
                'ang', 'anh', 'an', 'am', 'ach', 'ap', 'at',
                'ong', 'oc', 'op', 'ot',
                'ung', 'uc', 'up', 'ut',
                'inh', 'ich', 'ip', 'it',
                'eng', 'enh', 'em', 'en', 'ep', 'et',
                'ieu', 'eo',
                'oi',
                'ai', 'ao', 'au', 'ay',
                'uy',
                'a', 'e', 'i', 'o', 'u', 'y',
            ]
            for r in common:
                if t.endswith(r):
                    return r
            return t[-3:] if len(t) >= 3 else t

        def _validate_fixed_form(text: str, style_key: str) -> tuple[bool, str, dict]:
            """Return (ok, reason, meta). Meta may include rhyme_key."""
            lines = _split_lines(text)
            meta: dict = {'lines': lines}

            if style_key == 'tu_do':
                if len(lines) < 4 or len(lines) > 10:
                    return False, 'expected 4-10 lines for tu_do', meta

                # Rhythm guardrail: keep most lines within a reasonable syllable range.
                syllables = [_count_syllables_by_space(ln) for ln in lines]
                meta['syllables'] = syllables
                out_of_range = sum(1 for s in syllables if s < 5 or s > 12)
                if out_of_range >= max(2, len(lines) // 2):
                    return False, 'tu_do rhythm out of range', meta

                keys = [_rhyme_key_from_token(_last_token_normalized(ln)) for ln in lines]
                keys = [k for k in keys if k]
                if not keys:
                    return False, 'empty rhyme token', meta

                from collections import Counter
                c = Counter(keys)
                best_key, best_count = c.most_common(1)[0]
                meta['rhyme_key'] = best_key

                # Require consistent end-rhyme for most lines.
                need = max(3, int((len(lines) * 0.7) + 0.999))  # ceil(0.7 * n)
                if best_count < need:
                    return False, 'insufficient end-rhyme consistency', meta

                return True, '', meta

            if style_key == 'cau_doi':
                if len(lines) != 2:
                    return False, 'expected 2 lines for cau_doi', meta
                return True, '', meta

            if style_key == 'song_that_luc_bat':
                if len(lines) != 4:
                    return False, 'expected 4 lines for song_that_luc_bat', meta
                target = [7, 7, 6, 8]
                for i, t in enumerate(target):
                    if _count_syllables_by_space(lines[i]) != t:
                        return False, f'line {i+1} syllables != {t}', meta
                return True, '', meta

            if style_key == 'luc_bat':
                # Common lục bát can be 2 couplets (4 lines) or longer.
                if len(lines) not in (4, 6, 8):
                    return False, 'expected 4, 6, or 8 lines for luc_bat', meta
                if (len(lines) % 2) != 0:
                    return False, 'expected even number of lines for luc_bat', meta
                for i, ln in enumerate(lines):
                    need = 6 if (i % 2 == 0) else 8
                    if _count_syllables_by_space(ln) != need:
                        return False, f'line {i+1} syllables != {need}', meta
                return True, '', meta

            if style_key == 'that_ngon_tu_tuyet':
                if len(lines) != 4:
                    return False, 'expected 4 lines for that_ngon_tu_tuyet', meta
                for i, ln in enumerate(lines):
                    if _count_syllables_by_space(ln) != 7:
                        return False, f'line {i+1} syllables != 7', meta
                rhyme_idx = [0, 1, 3]
                keys = [_rhyme_key_from_token(_last_token_normalized(lines[i])) for i in rhyme_idx]
                keys = [k for k in keys if k]
                if not keys:
                    return False, 'empty rhyme token', meta
                # Require at least two of the rhyme positions share the same vần.
                from collections import Counter
                c = Counter(keys)
                best_key, best_count = c.most_common(1)[0]
                meta['rhyme_key'] = best_key
                if best_count < 2:
                    return False, 'insufficient end-rhyme consistency', meta
                return True, '', meta

            if style_key == 'that_ngon_bat_cu':
                if len(lines) != 8:
                    return False, 'expected 8 lines for that_ngon_bat_cu', meta
                for i, ln in enumerate(lines):
                    if _count_syllables_by_space(ln) != 7:
                        return False, f'line {i+1} syllables != 7', meta
                rhyme_idx = [0, 1, 3, 5, 7]
                keys = [_rhyme_key_from_token(_last_token_normalized(lines[i])) for i in rhyme_idx]
                keys = [k for k in keys if k]
                if not keys:
                    return False, 'empty rhyme token', meta
                from collections import Counter
                c = Counter(keys)
                best_key, best_count = c.most_common(1)[0]
                meta['rhyme_key'] = best_key
                # Require at least three of the five rhyme positions share the same vần.
                if best_count < 3:
                    return False, 'insufficient end-rhyme consistency', meta
                return True, '', meta

            # tu_do (no strict validation)
            return True, '', meta

        poem_text = (poem_text or '').strip()

        # Auto-repair for styles that need strong structure/rhyme guarantees.
        if poem_text and poem_style in {
            'tu_do',
            'luc_bat',
            'song_that_luc_bat',
            'that_ngon_tu_tuyet',
            'that_ngon_bat_cu',
            'cau_doi',
        }:
            ok, reason, meta = _validate_fixed_form(poem_text, poem_style)
            if not ok:
                rhyme_key = meta.get('rhyme_key')

                meter_check = {
                    'tu_do': 'KIỂM TRA THỂ: 4–8 dòng (ưu tiên 4–6 nếu prompt nói “ngắn”); mỗi dòng gọn, khoảng 6–10 tiếng để nhịp đều; gieo vần chân rõ ở phần lớn dòng.',
                    'luc_bat': 'KIỂM TRA THỂ: 4/6/8 dòng (số dòng chẵn); dòng 1/3/5/7 = 6 tiếng, dòng 2/4/6/8 = 8 tiếng (đếm theo các cụm tách bằng dấu cách).',
                    'song_that_luc_bat': 'KIỂM TRA THỂ: đúng 4 dòng, số tiếng lần lượt 7 / 7 / 6 / 8 (đếm theo các cụm tách bằng dấu cách).',
                    'that_ngon_tu_tuyet': 'KIỂM TRA THỂ: đúng 4 dòng, mỗi dòng đúng 7 tiếng (đếm theo các cụm tách bằng dấu cách).',
                    'that_ngon_bat_cu': 'KIỂM TRA THỂ: đúng 8 dòng, mỗi dòng đúng 7 tiếng (đếm theo các cụm tách bằng dấu cách).',
                    'cau_doi': 'KIỂM TRA THỂ: đúng 2 dòng (2 vế), cân đối độ dài.',
                }.get(poem_style, '')

                repair_system = (
                    'Bạn là biên tập viên thơ tiếng Việt. Hãy CHỈNH SỬA bài thơ bên dưới để tuân thủ đúng thể thơ và vần điệu. '
                    'Giữ ý nghĩa chúc Tết theo prompt người dùng, viết mượt, tự nhiên. '
                    'Nếu bài hiện tại sai thể (sai số tiếng/số dòng/vần), hãy VIẾT LẠI MỚI hoàn toàn theo đúng thể thơ thay vì sửa chắp vá. '
                    'Chống lỗi thường gặp: vỡ vần (mỗi dòng một vần), câu văn xuôi dài, lặp cụm từ (ví dụ “khắp nơi/muôn nơi”). '
                    'Ràng buộc: chỉ xuất ra bài thơ cuối cùng, mỗi câu một dòng, không tiêu đề, không giải thích.'
                )

                try:
                    for attempt in range(4):
                        extra_strict = (
                            'LƯU Ý: Bắt buộc đúng số tiếng từng dòng. Nếu dòng thiếu/dư tiếng, hãy thêm/bớt từ để đúng. '
                            'Tránh dấu câu rườm rà; ưu tiên câu chữ rõ ràng. '
                            'Giữ khoảng trắng chuẩn giữa các tiếng (mỗi tiếng cách nhau 1 dấu cách).'
                            if attempt == 1
                            else ''
                        )

                        tu_do_rhyme = (
                            (
                                f"THƠ TỰ DO: ưu tiên 4–6 dòng để nhịp gọn. "
                                f"Chọn 1 vần chủ đạo (đuôi như '{rhyme_key}' nếu đã có) và gieo vần chân xuyên suốt: "
                                f"ít nhất 3/4 dòng cùng vần (nếu 4 dòng) hoặc >=70% dòng cùng vần (nếu nhiều hơn). "
                                f"Tránh mỗi dòng một vần. Có thể đổi TỪ cuối nhưng phải giữ cùng VẦN (ví dụ '{rhyme_key}': vàng/rạng/sang/khang...). "
                                f"Không để 2 dòng liên tiếp kết thúc bằng đúng cùng một từ."
                            )
                            if poem_style == 'tu_do'
                            else ''
                        )

                        luc_bat_template = (
                            'MẪU LỤC BÁT (4 dòng):\n'
                            '(6 tiếng)\n'
                            '(8 tiếng)\n'
                            '(6 tiếng)\n'
                            '(8 tiếng)\n'
                            'Bạn có thể viết 6 hoặc 8 dòng nhưng phải luân phiên 6/8.'
                            if poem_style == 'luc_bat'
                            else ''
                        )

                        repair_user = (
                            f'PROMPT GỐC (để giữ ý):\n{prompt}\n\n'
                            f'THỂ THƠ BẮT BUỘC: {allowed_styles.get(poem_style, poem_style)}\n'
                            + (meter_check + '\n' if meter_check else '')
                            + (luc_bat_template + '\n' if luc_bat_template else '')
                            + (tu_do_rhyme + '\n' if tu_do_rhyme else '')
                            + (f"VẦN CHÂN BẮT BUỘC: các dòng gieo vần phải cùng VẦN (cùng đuôi vần như '{rhyme_key}'), gieo vần chân rõ ràng.\n" if rhyme_key else '')
                            + (extra_strict + '\n' if extra_strict else '')
                            + f'BÀI THƠ CẦN SỬA:\n{poem_text}'
                        )

                        chat2 = client.chat.completions.create(
                            model=model,
                            messages=[
                                {'role': 'system', 'content': repair_system},
                                {'role': 'user', 'content': repair_user},
                            ],
                            temperature=0.1,
                        )
                        repaired = (chat2.choices[0].message.content or '').strip()
                        if repaired:
                            ok2, _, _ = _validate_fixed_form(repaired, poem_style)
                            if ok2:
                                poem_text = repaired
                                break
                except Exception:
                    pass

                # Final fallback for tu_do: force a short 4-line poem with a single dominant rhyme.
                if poem_style == 'tu_do':
                    ok_final, _, meta_final = _validate_fixed_form(poem_text, poem_style)
                    if not ok_final:
                        rk = (rhyme_key or meta_final.get('rhyme_key') or '').strip()
                        suggestions_map = {
                            'ang': ['vàng', 'sang', 'khang', 'rạng', 'trang', 'an khang'],
                            'an': ['an', 'bình an', 'vẹn toàn', 'an nhàn'],
                            'am': ['ấm', 'đằm', 'thắm'],
                            'ien': ['hiên', 'yên', 'duyên', 'thiên', 'tiên', 'miên'],
                            'ong': ['thông', 'trong', 'hồng', 'đông'],
                            'ay': ['ngày', 'đầy', 'say', 'bay'],
                            'a': ['hoa', 'nhà', 'xa', 'ta'],
                            'oi': ['vui', 'thôi', 'mới', 'tươi'],
                            'em': ['êm', 'thêm', 'mềm'],
                            'en': ['bền', 'quen', 'khen'],
                        }
                        suggested = suggestions_map.get(rk, [])
                        suggested_str = ', '.join(suggested) if suggested else ''

                        force_system = (
                            'Bạn là thi sĩ Việt Nam. Hãy VIẾT MỚI một bài thơ Tết THƠ TỰ DO thật gọn và đã tai. '
                            'Bắt buộc: đúng 4 dòng; mỗi dòng 6–10 tiếng; gieo vần chân rõ; không dấu câu ở cuối dòng; tránh lặp cụm từ. '
                            'Chỉ trả về bài thơ, mỗi câu một dòng.'
                        )
                        force_user = (
                            f'PROMPT: {prompt}\n'
                            + (f"VẦN CHỦ ĐẠO: dùng vần '-{rk}' cho cả 4 dòng. " if rk else '')
                            + (f"GỢI Ý TỪ KẾT VẦN (không bắt buộc, tránh lặp từ): {suggested_str}. " if suggested_str else '')
                            + 'Hãy để 4 dòng kết thúc bằng 4 từ khác nhau nhưng cùng vần.'
                        )

                        try:
                            for _ in range(2):
                                chat3 = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {'role': 'system', 'content': force_system},
                                        {'role': 'user', 'content': force_user},
                                    ],
                                    temperature=0.6,
                                )
                                candidate = (chat3.choices[0].message.content or '').strip()
                                if candidate:
                                    ok3, _, _ = _validate_fixed_form(candidate, poem_style)
                                    if ok3:
                                        poem_text = candidate
                                        break
                        except Exception:
                            pass
        if poem_text:
            title = poem_text.splitlines()[0].strip() if poem_text.splitlines() else ''
            if not title:
                title = 'Thơ chúc Tết'
            _add_project(
                'poem',
                title,
                {
                    'prompt': _cap_text(prompt, 600),
                    'poem': _cap_text(poem_text, 1400),
                    'poem_style': poem_style,
                    'prompt_truncated': bool(prompt_truncated),
                },
            )

        return jsonify({'status': 'success', 'poem': poem_text, 'model': model, 'poem_style': poem_style})
    except Exception as e:
        return jsonify({'status': 'error', 'error': f'OpenAI request failed: {e}'}), 500


@app.route('/api/generate-image', methods=['POST'])
def api_generate_image():
    """API: Generate an image from a single prompt. Returns a data URL."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        return jsonify({'status': 'error', 'error': 'Missing prompt'}), 400

    provider = (os.environ.get('IMAGE_PROVIDER') or 'stability').strip().lower()
    if provider == 'gemini':
        payload, err = _generate_image_with_gemini(prompt)
    elif provider == 'openai':
        payload, err = _generate_image_with_openai(prompt)
    else:
        # Stability works best with English prompts; translate via OpenAI when needed.
        translated_prompt = prompt
        if _should_translate_to_english(prompt):
            translated_prompt, tr_err = _translate_to_english_with_openai(prompt)
            if tr_err:
                return jsonify({'status': 'error', 'error': tr_err}), 500

        payload, err = _generate_image_with_stability(translated_prompt)
        if payload is not None:
            payload.setdefault('original_prompt', prompt)
            payload.setdefault('translated_prompt', translated_prompt)

    if err:
        return jsonify({'status': 'error', 'error': err}), 500

    persisted_url = None
    if isinstance(payload, dict):
        img_field = payload.get('image')
        if isinstance(img_field, str) and img_field.startswith('data:image/'):
            persisted_url = _save_data_url_image_to_static(img_field)
            if persisted_url:
                payload['image_url'] = persisted_url
        else:
            url_field = payload.get('image_url')
            if isinstance(url_field, str) and url_field:
                persisted_url = url_field

    # Save a lightweight project (do NOT store base64 image data in cookie session)
    _add_project(
        'image',
        'Ảnh chúc Tết',
        {
            'prompt': _cap_text(prompt, 600),
            'image_url': _cap_text(persisted_url or '', 600),
            'provider': _cap_text((payload or {}).get('provider') if isinstance(payload, dict) else '', 60),
            'model': _cap_text((payload or {}).get('model') if isinstance(payload, dict) else '', 60),
        },
    )
    return jsonify({'status': 'success', **(payload or {})})


@app.route('/api/generate-music', methods=['POST'])
def api_generate_music():
    """API: Generate music via Suno provider. Optionally remake lyrics via OpenAI before sending."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        return jsonify({'status': 'error', 'error': 'Missing prompt'}), 400

    trimmed_prompt = _trim_to_max_words(prompt)
    prompt_truncated = trimmed_prompt != prompt
    prompt = trimmed_prompt

    # Optional per-request provider fields (Kie.ai Suno API custom mode)
    title = (data.get('title') or '').strip() or None
    style = (data.get('style') or '').strip() or None
    model = (data.get('model') or '').strip() or None
    custom_mode = data.get('custom_mode')
    instrumental = data.get('instrumental')

    remake_lyrics = bool(data.get('remake_lyrics'))
    remake_length = (data.get('remake_length') or '').strip() or None
    remake_note = (data.get('remake_note') or '').strip() or None
    lyrics = (data.get('lyrics') or '').strip() or None

    remix = None
    if remake_lyrics:
        remix, err = _generate_lyrics_with_openai(prompt, length=remake_length, note=remake_note)
        if err:
            return jsonify({'status': 'error', 'error': err}), 500
        lyrics = (remix or {}).get('lyrics') or lyrics

    # In Kie custom mode, we often treat prompt as exact lyrics. If user remade lyrics,
    # use that as the prompt too (so the sung lyrics match the remade version).
    prompt_to_provider = lyrics or prompt

    extra_fields: dict = {}
    if title:
        extra_fields['title'] = title
    if style:
        extra_fields['style'] = style
    if model:
        extra_fields['model'] = model
    if custom_mode is not None:
        extra_fields['custom_mode'] = bool(custom_mode)
    if instrumental is not None:
        extra_fields['instrumental'] = bool(instrumental)

    music, err = _call_suno_generate(prompt=prompt_to_provider, lyrics=lyrics, extra_fields=extra_fields)
    if err:
        return jsonify({'status': 'error', 'error': err}), 500

    resp = {
        **(music or {}),
        'remake_lyrics': remake_lyrics,
        'prompt_truncated': prompt_truncated,
    }
    if lyrics:
        resp['lyrics'] = lyrics
    if remix and isinstance(remix, dict):
        resp['lyrics_provider'] = remix.get('provider')
        resp['lyrics_model'] = remix.get('model')

    # Save a lightweight project
    proj_title = title or ''
    if not proj_title:
        first_line = (prompt or '').splitlines()[0].strip() if (prompt or '').splitlines() else ''
        proj_title = first_line[:60] if first_line else 'Bài nhạc'
    _add_project(
        'music',
        proj_title,
        {
            'prompt': _cap_text(prompt, 1200),
            'title': _cap_text(title or proj_title, 120),
            'style': _cap_text(style or '', 120),
            'lyrics': _cap_text(resp.get('lyrics') if isinstance(resp, dict) else '', 1800),
            'audio_url': _cap_text((music or {}).get('audio_url') if isinstance(music, dict) else '', 600),
            'remake_lyrics': bool(remake_lyrics),
            'prompt_truncated': bool(prompt_truncated),
        },
    )

    return jsonify({'status': 'success', **resp})


# =============================================
# API ENDPOINTS - Tích hợp AI (Mẫu)
# =============================================

@app.route('/api/poem-to-music', methods=['POST'])
def poem_to_music():
    """
    API: Chuyển thơ thành âm nhạc
    Input: {poem: "text"}
    Output: {audio_url: "...", analysis: {...}}
    """
    data = request.json
    poem = data.get('poem', '')
    
    # TODO: Tích hợp AI model thực tế
    response = {
        'status': 'success',
        'audio_url': '/static/generated/sample-music.mp3',
        'analysis': {
            'tone': 'Trầm buồn, hoài niệm',
            'tempo': 'Chậm (Adagio)',
            'key': 'D minor',
            'instruments': ['Đàn tranh', 'Sáo trúc', 'Piano']
        }
    }
    
    return jsonify(response)

@app.route('/api/poem-to-image', methods=['POST'])
def poem_to_image():
    """
    API: Chuyển thơ thành hình ảnh
    Input: {poem: "text"}
    Output: {image_url: "...", description: "..."}
    """
    data = request.json
    poem = data.get('poem', '')
    
    # TODO: Tích hợp Stable Diffusion/DALL-E
    response = {
        'status': 'success',
        'image_url': '/static/generated/sample-image.jpg',
        'description': 'Cảnh sắc xuân với mai vàng và đèn lồng đỏ',
        'prompt_used': 'Vietnamese traditional spring festival, golden plum blossoms...'
    }
    
    return jsonify(response)

@app.route('/api/fashion-redesign', methods=['POST'])
def fashion_redesign():
    """
    API: Tái thiết kế trang phục truyền thống
    Input: {style: "...", elements: [...]}
    Output: {design_url: "...", description: "..."}
    """
    data = request.json
    
    response = {
        'status': 'success',
        'design_url': '/static/generated/fashion-design.jpg',
        'description': 'Áo dài hiện đại kết hợp họa tiết rồng phượng',
        'elements': ['Áo dài', 'Họa tiết long phụng', 'Màu đỏ vàng kim']
    }
    
    return jsonify(response)

@app.route('/api/chat-heritage', methods=['POST'])
def chat_heritage():
    """
    API: Chatbot đối thoại về văn học
    Input: {message: "...", context: [...]}
    Output: {reply: "...", sources: [...]}
    """
    data = request.json
    message = data.get('message', '')
    
    # TODO: Tích hợp LLM (GPT/Claude) với knowledge base
    response = {
        'status': 'success',
        'reply': 'Thơ Xuân là dòng thơ truyền thống của Việt Nam, thường được sáng tác vào dịp Tết Nguyên Đán...',
        'sources': [
            'Thi pháp Việt Nam - Hoàng Xuân Nhị',
            'Lịch sử văn học Việt Nam - Viện Văn học'
        ]
    }
    
    return jsonify(response)

@app.route('/api/generate-wish', methods=['POST'])
def generate_wish():
    """
    API: Tạo lời chúc Tết phong cách cổ điển
    Input: {recipient: "...", tone: "...", keywords: [...]}
    Output: {wish: "...", calligraphy_style: "..."}
    """
    data = request.json
    
    response = {
        'status': 'success',
        'wish': 'Xuân về muôn nẻo sum vầy,\nPhúc lành tràn ngập trong ngày đầu năm.',
        'style': 'Song thất lục bát',
        'calligraphy_style': 'Chữ Hán Nôm',
        'image_url': '/static/generated/wish-calligraphy.jpg'
    }
    
    return jsonify(response)

# =============================================
# ERROR HANDLERS
# =============================================

@app.errorhandler(404)
def not_found(error):
    # Keep this lightweight (no extra templates required)
    return 'Not Found', 404

@app.errorhandler(500)
def internal_error(error):
    # Keep this lightweight (no extra templates required)
    return 'Internal Server Error', 500

# =============================================
# DEVELOPMENT SERVER
# =============================================

if __name__ == '__main__':
    debug = True
    if debug:
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=debug, host='0.0.0.0', port=5000)
