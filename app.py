"""
SẮC XUÂN VĂN HỌC - Flask Application
Nền tảng AI kết nối di sản văn học Việt Nam
"""

from flask import Flask, render_template, jsonify, request
import os

# Load local .env if present (keeps secrets out of code)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
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


@app.route('/api/generate-poem', methods=['POST'])
def api_generate_poem():
    """API: Generate a Tet wish poem from a single prompt."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        return jsonify({'status': 'error', 'error': 'Missing prompt'}), 400

    client, err = _get_openai_client()
    if err:
        return jsonify({'status': 'error', 'error': err}), 500

    model = os.environ.get('OPENAI_TEXT_MODEL', 'gpt-4o-mini')
    system = (
        'Bạn là một trợ lý tiếng Việt chuyên viết thơ chúc Tết. '
        'Hãy tạo một bài thơ chúc Tết ngắn gọn, tự nhiên, dễ đọc, phù hợp để gửi tin nhắn. '
        'Trả về đúng nội dung bài thơ, không thêm tiêu đề, không thêm giải thích.'
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

    client, err = _get_openai_client()
    if err:
        return jsonify({'status': 'error', 'error': err}), 500

    model = os.environ.get('OPENAI_IMAGE_MODEL', 'gpt-image-1')
    size = os.environ.get('OPENAI_IMAGE_SIZE', '1024x1024')

    try:
        img = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            response_format='b64_json',
        )
        b64 = img.data[0].b64_json
        data_url = f'data:image/png;base64,{b64}'
        return jsonify({'status': 'success', 'image': data_url, 'model': model, 'size': size})
    except Exception as e:
        return jsonify({'status': 'error', 'error': f'OpenAI image request failed: {e}'}), 500


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
