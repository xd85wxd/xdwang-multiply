import torch
import os
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

MODEL_PATH = "/home/ehpc/data/多模态/modelscope_models/Qwen/Qwen2___5-VL-7B-Instruct"

print("Loading Qwen2.5-VL...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Qwen2.5-VL loaded.")


def ask_image(image_path, history):

    
    messages = []

    first_user = True

    for msg in history:
        if msg["role"] == "user":
            if first_user:
                messages.append(
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "image":image_path
                                },
                                {
                                    "type":"text",
                                    "text":msg["content"]
                                }
                            ]
                        }
                    )
                first_user = False
            else:
                messages.append(
                        {
                            "role":"user",
                            "content":msg["content"]
                        }
                    )
        else:
            messages.append(
                    {
                        "role":"assistant",
                        "content":msg["content"]
                    }
                )
    print("=" * 50)
    print(messages)
    print("=" * 50)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]

    return response
def ask_vqa(image_path, question):

    messages = [
        {
            "role": "system",
            "content":
                "You are a Visual Question Answering model. "
                "Answer with only a short answer. "
                "Do not explain. "
                "Use one or two words whenever possible."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]
    del inputs
    del generated_ids
    del generated_ids_trimmed
    del image_inputs
    del video_inputs    
    gc.collect()
    torch.cuda.empty_cache()
    return response.strip()
def ask_textvqa(image, question):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Visual Question Answering model. "
                "Answer briefly, no explanation."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image   
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]

    return response.strip()