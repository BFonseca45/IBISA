import numpy as np
import torch
from tqdm import tqdm

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
np.random.seed(10)

class EvalMetric():
    def __init__(self, model) -> None:
        self.model = model

    def get_tiles(self, saliency, STEP=8):
        self.STEP = STEP
        importance = []
        coords = []

        for i in range(0, len(saliency[0,:]), self.STEP):
            for n in range(0, len(saliency[:,0]), self.STEP):
                aux = np.sum(saliency[i:i+self.STEP, n:n+self.STEP])
                importance.append(aux)
                coords.append([i, n])

        return coords, importance

    def sort_coordenates(self, importance, coords, reverse = True):
        importance_sorted = np.unique(importance)
        importance_sorted = sorted(importance_sorted, reverse=reverse) # Change this line to remove the least important pixels first

        coords_sorted = []
        for t in range(len(importance_sorted)):
            ind = np.where(importance == importance_sorted[t])
            for p in range(len(ind[0])):
                coords_sorted.append(coords[ind[0][p]])

        return coords_sorted

    def compute_sim(self, image, y):
        preds = self.model(image)
        preds = torch.nn.functional.softmax(preds, dim=1)

        return preds[0,y].detach().cpu().numpy()

    def first_score(self, image):
        score = []
        preds = self.model(image)
        y = torch.argmax(preds, dim=1)

        similarity = self.compute_sim(image = image, y = y)
        score.append(similarity)

        return score, y

    def compute_score(self, img, saliency, metric):

        coords, importance = self.get_tiles(saliency=saliency)

        if metric == 'MoRF':
            mask = torch.ones_like(img, dtype=torch.float32)
            ocl = 0
            coords_sorted = self.sort_coordenates(coords=coords, importance=importance, reverse=True)
            score, label = self.first_score(image=img)

        elif metric == 'LeRF':
            mask = torch.ones_like(img, dtype=torch.float32)
            ocl = 0
            coords_sorted = self.sort_coordenates(coords=coords, importance=importance, reverse=False)
            score, label = self.first_score(image=img)

        for i in range(len(coords_sorted) - 1):
            idx = coords_sorted[i][0]
            idy = coords_sorted[i][1]
            mask[:,:,idx:idx+self.STEP, idy:idy+self.STEP] = ocl

            # masking the input image
            img_in = img * mask
            img_in = img_in.to(device)

            similarity = self.compute_sim(image=img_in, y=label)
            score.append(similarity)

        score = np.array(score, dtype=np.float32)
        score = (score - min(score))/(max(score) - min(score))

        x = np.linspace(start=0, stop=1, num=len(score))
        area = np.trapz(np.squeeze(score),x)

        return score, area

    def gauss_deg(self, saliency, scale_shape):
        imgg = self.img.detach().cpu().numpy()

        similarity0 = self.compute_sim(image=self.img)

        noise = np.random.randn(1,3,scale_shape,scale_shape)*np.std(imgg) + np.mean(imgg)

        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        saliency_norm = np.stack((saliency_norm, saliency_norm, saliency_norm))
        saliency_norm = saliency_norm[None,...]

        img_in = saliency_norm*imgg + (1 - saliency_norm)*noise
        img_in = torch.Tensor(img_in)
        img_in = img_in.to(device)

        similarity = self.compute_sim(image=img_in)
        ratio = (similarity + 1e-6)/(similarity0 + 4e-6)

        return img_in, np.clip(ratio, 0, 1)

    def pointing_game(self, box_coord, saliency):
        ind = np.unravel_index(np.argmax(saliency), saliency.shape)
        if (ind[1] > box_coord[0] and ind[1] < box_coord[2]) and (ind[0] > box_coord[1] and ind[0] < box_coord[3]):
            return 1, ind
        else:
            return 0, ind
        
def kl_div_uni(x1, x2):
    mu_x1 = torch.mean(x1)
    mu_x2 = torch.mean(x2)

    std_x1 =torch.std(x1)
    std_x2 = torch.std(x2)

    dkl = torch.log(std_x2/std_x1) + (std_x1**2 + (mu_x1 - mu_x2)**2)/(2*std_x2**2) - 0.5

    return dkl

def mutual_information(hgram):

    """ 
    
    Mutual information for joint histogram

    """

    # Convert bins counts to probability values

    pxy = hgram / float(np.sum(hgram))

    px = np.sum(pxy, axis=1) # marginal for x over y

    py = np.sum(pxy, axis=0) # marginal for y over x

    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px_py 2D arrays

    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def brier_score(target_hot, probs):
    BS = []
    for prob in probs:
        prob = prob[0]
        brier = (target_hot - prob)**2
        brier = np.sum(brier)/len(prob)
        BS.append(brier)
    
    return BS