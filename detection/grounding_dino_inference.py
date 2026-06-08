import argparse
import os
import json
import torch
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict

# # ======================================================
# # FIRST YOU HAVE TO DOWNLOAD MANUALLY THE WEIGHTS. RUN THIS UN YOUR COMMAND LINE
# # ======================================================
# mkdir weights
# cd weights
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TEXT_PROMPT = 'ulcer . foot ulcer . diabetic foot ulcer .'  # separate prompts with .
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

CONFIG_PATH = 'weights/GroundingDINO_SwinB.cfg.py'
WEIGHTS_PATH = 'weights/groundingdino_swinb_cogcoor.pth'


def run_inference(images_dir, output_file,
                  box_threshold=BOX_THRESHOLD,
                  text_threshold=TEXT_THRESHOLD,
                  text_prompt=TEXT_PROMPT):

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    results = {}

    image_files = [f for f in os.listdir(images_dir)
                   if f.endswith(('.jpg', '.jpeg', '.png'))]

    for fname in tqdm(image_files, desc='Running GroundingDINO inference'):
        img_path = os.path.join(images_dir, fname)
        image_source, image = load_image(img_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=DEVICE
        )

        results[fname] = {
            'boxes': boxes.cpu().numpy().tolist(),
            'scores': logits.cpu().numpy().tolist(),
            'phrases': phrases
        }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Resultados guardados en {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str,
                        default='results/grounding_dino_predictions.json')
    parser.add_argument('--box_threshold', type=float, default=BOX_THRESHOLD)
    parser.add_argument('--text_threshold', type=float, default=TEXT_THRESHOLD)
    parser.add_argument('--text_prompt', type=str, default=TEXT_PROMPT)
    args = parser.parse_args()

    run_inference(
        images_dir=args.images_dir,
        output_file=args.output_file,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        text_prompt=args.text_prompt
    )