import config
from config import *


class CustomDataset(Dataset):
    def __init__(self, _dir, transform=None):
        self.imgs_pt = []
        self.labels_pt = []
        for root, dirs, files in os.walk(_dir):
            for file in files:
                path = os.path.join(root, file)
                img_pt = torch.tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))/255.0
                img_pt = img_pt
                label_pt = torch.tensor(config.LABELS.index(root.split('/')[-1]))
                label_pt = label_pt
                self.labels_pt.append(label_pt)
                self.imgs_pt.append(img_pt)
        self.labels_pt = torch.tensor(self.labels_pt)
        self.imgs_pt = torch.stack(self.imgs_pt)
        self.transforms = transform

    def __getitem__(self, index):
        data = self.imgs_pt[index]
        n, m, c = data.size()
        data = torch.reshape(data, (c, n, m))
        if self.transforms is not None:
            tensor2pil = transforms.ToPILImage()
            data = tensor2pil(data)
            data = self.transforms(data)

        return data.to(config.DEVICE), self.labels_pt[index].to(config.DEVICE)

    def __len__(self):
        return len(self.labels_pt)
