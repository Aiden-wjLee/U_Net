import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from UNet import UNet
import glob
import cv2
import matplotlib.pyplot as plt

class UNet_mIoU_check():
    def __init__(self, device, model_path, img_folder, mask_folder, show_=False):
        self.device = device
        self.model_path = model_path
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.net = UNet().to(device)  # UNet 모델 구조를 정의해야 합니다.
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()
        self.mean_IoU = 0  # mIoU 변수명을 mean_IoU로 변경했습니다.
        self.mIoU_arr = []
        self.transform = transforms.Compose([transforms.ToTensor()])  # 필요한 전처리를 추가합니다.

    def load_image(self, img_path, mask_path):
        img = Image.open(img_path).convert('RGB')
        img = cv2.resize(np.array(img), (256, 256))
        img = self.transform(Image.fromarray(img))
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def inference(self, img):
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():  # 추론 시에는 gradient 계산이 필요 없습니다.
            output = self.net(img)
        #최대값을 갖는 인덱스를 찾아줍니다.
        output = output.argmax(dim=1)
        if show_:
            cv2.imwrite("output.jpg", output.squeeze().cpu().numpy()*255)
            cv2.waitKey(0)
        return output.cpu().squeeze().numpy()  # numpy 배열로 변환

    def calculate_mIoU(self, output, label):
        intersection = np.logical_and(label, output)
        union = np.logical_or(label, output)
        iou_score = np.sum(intersection) / np.sum(union)
        self.mIoU_arr.append(iou_score)
        self.mean_IoU = np.mean(self.mIoU_arr)
        return self.mean_IoU

    def evaluate(self):
        img_paths = glob.glob(self.img_folder + '/*.jpg') + glob.glob(self.img_folder + '/*.png')
        mask_paths = glob.glob(self.mask_folder + '/*.jpg') + glob.glob(self.mask_folder + '/*.png')
        assert len(img_paths) == len(mask_paths), "The number of images and masks must match"

        # 'x.jpg' 에서 x의 숫자대로 정렬
        img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[-2]))
        mask_paths = sorted(mask_paths, key=lambda x: int(os.path.basename(x).split('.')[-2]))


        all_iou_scores = []  # Store individual IoU scores
        for img_path, mask_path in zip(img_paths, mask_paths):
            img, mask = self.load_image(img_path, mask_path)
            output = self.inference(img)
            iou = self.calculate_mIoU(output, mask)
            all_iou_scores.append(iou)  # Store IoU for this image-mask pair
            print(f'IoU for {img_path}: {iou}')

        self.mean_IoU = np.mean(all_iou_scores)  # Calculate mean IoU here
        print(f'Mean IoU for dataset: {self.mean_IoU}')

if __name__ == '__main__':
    print("s")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './BEST_green_real_floor3.pth'
    img_folder = '/media/lee/90182A121829F83C/Papers/Track-Anything_/result/img/kros_miou1'
    mask_folder = '/media/lee/90182A121829F83C/Papers/Track-Anything_/result/mask/kros_miou1'
    show_= True
    miou_check=UNet_mIoU_check(device, model_path, img_folder, mask_folder, show_)
    miou_check.evaluate()
