from PIL import Image
import re
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from collections import Counter
from src.utils import load_pickle

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

def normalize(text):
    text = text.strip().lower()
    text = re.sub(r'^\W+', '', text)  # remove non-word characters at start
    text = re.sub(r'\W+$', '', text)  # remove non-word characters at end
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text

def classify_images(image_list, category_prompts):
    stats = Counter()
    normalized_categories = {normalize(cat): cat for cat in category_prompts}

    for image in image_list:
        prompt_text = (
            "Classify the image to one of the following prompts: "
            + ", ".join(category_prompts)
            + ". Return exact prompt without any additional text. Return 'UNCLASSIFIED' if the image does not match any prompt."
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        decoded = processor.decode(output[0][2:], skip_special_tokens=True).strip().lower()
        # print(decoded)

        match = re.search(r'assistant:\s*(.*)', decoded, re.IGNORECASE)
        if match:
            response = normalize(match.group(1))
        else:
            response = normalize(decoded)

        if response in normalized_categories:
            matched = normalized_categories[response]
        else:
            matched = "UNCLASSIFIED"

        stats[matched] += 1
        print(f"Image -> {matched}")
    return stats
