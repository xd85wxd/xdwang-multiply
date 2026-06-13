import os

from flask import Flask
from flask import render_template
from flask import request

from model import ask_image

from PIL import Image

app = Flask(__name__)

chat_history = []

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# 当前图片缓存
current_image_name = None


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/ask", methods=["POST"])
def ask():

    global chat_history
    global current_image_name

    print("Receive Request")

    image = request.files["image"]
    question = request.form["question"]

    print("Image:", image.filename)
    print("Question:", question)

    image_path = os.path.join(
        UPLOAD_FOLDER,
        image.filename
    )

    image.save(image_path)

    print("=" * 60)
    print("Saved File:", image_path)
    print(
        "File Size:",
        os.path.getsize(image_path),
        "bytes"
    )

    try:
        img = Image.open(image_path)
        print("Resolution:", img.size)
    except Exception as e:
        print("PIL Error:", e)

    print("=" * 60)

    # ==================================
    # 新图片
    # ==================================
    if image.filename != current_image_name:

        print("New Image Detected")

        current_image_name = image.filename

        # 换图后清空历史
        chat_history = []

    # ==================================
    # 记录用户问题
    # ==================================
    chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    # ==================================
    # Qwen2.5-VL问答
    # ==================================
    answer = ask_image(
        image_path,
        chat_history
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return {
        "answer": answer
    }


@app.route("/new_chat", methods=["POST"])
def new_chat():

    global chat_history
    global current_image_name

    chat_history = []
    current_image_name = None

    return {
        "status": "ok"
    }


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )