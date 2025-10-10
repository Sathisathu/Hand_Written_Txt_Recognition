import torch
from PIL import Image
import torchvision.transforms as T
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.htr_model import HTRModel
from data_loader.dataset import IAMDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelServer:
    def __init__(self):
        print("Initializing ModelServer...")

        dataset = IAMDataset(
            images_dir="data/lines",
            labels_file="data/labels.txt",
            img_height=64,
            max_width=256
        )
        self.idx_to_char = dataset.idx_to_char
        self.blank_idx = 0
        num_classes = len(dataset.chars) + 1


        self.model = HTRModel(num_classes=num_classes).to(DEVICE)
        ckpt_path = "checkpoints/htr_model_best.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def preprocess_image(self, img_path, img_height=64, max_width=256):
        img = Image.open(img_path).convert("L")  # grayscale
        w, h = img.size
        new_w = int(w * (img_height / h))
        new_w = min(new_w, max_width)
        img = img.resize((new_w, img_height))

        new_img = Image.new("L", (max_width, img_height), color=255)
        new_img.paste(img, (0, 0))

        tensor = self.transform(new_img).unsqueeze(0)  # [1, 1, H, W]
        return tensor

    def ctc_decode(self, output_probs):
        output = output_probs.argmax(2).squeeze(1).detach().cpu().numpy()
        prev = -1
        text = ""
        for idx in output:
            if idx != prev and idx != self.blank_idx:
                text += self.idx_to_char.get(idx, "")
            prev = idx
        return text

    def predict(self, img_path, predicted_text=None):
        img_tensor = self.preprocess_image(img_path).to(DEVICE)
        with torch.no_grad():
            output = self.model(img_tensor)              # [B, W, C]
            output = output.permute(1, 0, 2)             # [W, B, C]
            text = self.ctc_decode(output)
        predicted_text = text.replace("|", " ")
        return predicted_text

model_server = ModelServer()
