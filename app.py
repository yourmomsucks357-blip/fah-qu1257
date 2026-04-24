import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import requests
from io import BytesIO

MODEL_NAME = "huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
print("Ready.")

def respond(message, image, history):
    content = []

    if image is not None:
        content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": message})

    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]})

    messages.append({"role": "user", "content": content})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=2048)
    response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

with gr.Blocks(theme=gr.themes.Base(), title="FAH-QU // Claude 4.7 Opus Abliterated") as demo:
    gr.Markdown("# FAH-QU\n### Qwen3.6-35B · Claude 4.7 Opus Abliterated · Vision + Text")

    chatbot = gr.Chatbot(height=400)
    image_input = gr.Image(type="pil", label="Image (optional)")
    msg = gr.Textbox(placeholder="Type your message...", label="Message")
    send_btn = gr.Button("SEND", variant="primary")
    clear_btn = gr.Button("CLEAR")

    state = gr.State([])

    def user_submit(message, image, history):
        if not message.strip():
            return history, history, "", None
        response = respond(message, image, history)
        history.append((message, response))
        return history, history, "", None

    send_btn.click(user_submit, [msg, image_input, state], [chatbot, state, msg, image_input])
    msg.submit(user_submit, [msg, image_input, state], [chatbot, state, msg, image_input])
    clear_btn.click(lambda: ([], [], "", None), None, [chatbot, state, msg, image_input])

demo.launch()
