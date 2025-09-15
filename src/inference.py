import argparse, os, torch, cv2
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from model import build_model

ID2NAME = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited',
    17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right',
    21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

def build_tf(img_size=224):
    return Compose([
        Resize(img_size, img_size),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

@torch.no_grad()
def predict_image(model, img_path, tf, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = tf(image=img)['image'].unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    cls = int(prob.argmax())
    return cls, float(prob.max())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best.pt')
    parser.add_argument('--img', type=str, required=True)   # path to input image
    parser.add_argument('--num_classes', type=int, default=43)
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location='cpu')
    backbone = ckpt['backbone']; img_size = ckpt['img_size']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.num_classes, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    tf = build_tf(img_size)
    cls, conf = predict_image(model, args.img, tf, device)
    print(f"Pred: {ID2NAME.get(cls, cls)} (conf={conf:.3f})")

if __name__ == '__main__':
    main()