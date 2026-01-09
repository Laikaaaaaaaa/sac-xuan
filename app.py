"""
SẮC XUÂN VĂN HỌC - Flask Application
Nền tảng AI kết nối di sản văn học Việt Nam
"""

from flask import Flask, render_template, jsonify, request
import os
import json
import time
import base64

# Load local .env if present (keeps secrets out of code)
try:
    from dotenv import load_dotenv  # type: ignore

    # In local dev, prefer .env values over inherited shell env vars.
    load_dotenv(override=True)
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

# =============================================
# ROUTES - Các trang chính
# =============================================

@app.route('/')
def home():
    """Trang chủ - Landing cho AI tạo thơ chúc Tết"""
    return render_template('home.html')


@app.route('/main')
@app.route('/main/')
def main():
    """Trang tạo thơ chúc Tết"""
    return render_template('main.html')


@app.route('/image')
@app.route('/image/')
def image():
    """Trang tạo ảnh chúc Tết từ prompt"""
    return render_template('image.html')


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
    if not prompt:
        return jsonify({'status': 'error', 'error': 'Missing prompt'}), 400

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

    model = os.environ.get('OPENAI_TEXT_MODEL', 'gpt-4o-mini')
    system = (
        'Bạn là một thi sĩ tiếng Việt chuyên sáng tác thơ chúc Tết. '
        'Nhiệm vụ: viết một bài thơ chúc Tết NGẮN, HAY, có VẦN và NHỊP tự nhiên; ngôn ngữ trong sáng, giàu hình ảnh, gợi không khí sum vầy, an khang. '
        'Ưu tiên cách viết mộc mà sang, tránh sáo rỗng; hạn chế lặp từ; dùng phép gieo vần chân hoặc vần liền cho mượt. '
        'Nếu người dùng cung cấp người nhận/từ khóa, hãy lồng vào khéo léo (không liệt kê). '
        'Ràng buộc: chỉ trả về NỘI DUNG BÀI THƠ (4–8 dòng), không tiêu đề, không giải thích, không chào hỏi.'
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

        return jsonify({'status': 'success', 'poem': poem_text.strip(), 'model': model})
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
    return jsonify({'status': 'success', **(payload or {})})


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
