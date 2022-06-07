import torch

from src.utils import util_data


# BASE class to pass to PSO, implement methods:
# fitness,
# load_gan
# load_cnn
# particles_to_img

class Discovery:

    def __init__(self, iid_class, model_gan, model_cnn, device, control_pso_fitness='optimize_out_training', iid_classes=None, obj_fun_threshold=0.):
        self.model_gan = model_gan
        self.model_cnn = model_cnn
        #self.model_encoder = model_encoder
        self.device = device
        self.threshold = obj_fun_threshold
        self.iid_class = iid_class

        # todo choose how to optimize the cnn
        self.control_pso_fitness = control_pso_fitness # parameter active only for pso_inverter. It controls if we want to optimize the particles
        # positions to make the cnn to say "In Training" (posterior probability near 0) or "Out of Training" (posteriore probability near 1).
        # In the first case (optimize_in_training is True) we search to move the latent representations of new patients near one of the patient already
        # discovered in latent space via pso framework. In the second case (optimize_in_training i False) we are searching for diverse position with
        # respect to the in distribution patients.

        if iid_classes is not None:
            self.class_to_idx = {c: i for i, c in enumerate(sorted(iid_classes))}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def particles_to_img(self,dim_space, pos):

        if pos.ndim != 4:
            #print(f"Unsupported input dimension: Expected 4D (batched) input but got input of size: {pos.shape}! Let's add dimension!")
            for _ in range(4 - pos.ndim):
                pos = torch.unsqueeze(pos, dim=0)
            pos = pos.view([1, dim_space, 1, 1])

        if pos.dtype == torch.float64:
            #print( f"Unsupported input type: Expected float32 but got {pos.dtype}! Let's change Tensor dtype!")
            pos = pos.to(torch.float32)

        self.model_gan.eval()
        with torch.no_grad():
            img = self.model_gan(pos)
        img_rescaled = util_data.rescale_torch(img)

        return img_rescaled, img

    def fitness(self, dim_space, pos, eps=0.1):
        dtype = type(pos)
        if isinstance(pos, torch.Tensor):
            pass
        else:
            #print(f'Unsupported input type `{dtype}`! Convert to Torch Tensor!')
            pos = torch.from_numpy(pos)

        pos = pos.to(self.device)
        img_rescaled, img = self.particles_to_img(dim_space, pos)

        assert torch.max(img_rescaled) <= 1.0
        assert torch.min(img_rescaled) >= 0.0

        self.model_cnn.eval()
        with torch.no_grad():
            output = self.model_cnn(img_rescaled.float())
            if output.ndim > 1:
                output = torch.nn.functional.softmax(output, dim=1, dtype=torch.float32)[0]
                if output.shape[0] > 2:
                    output = output[self.class_to_idx[self.iid_class]]
                else:
                    output = output[1]
                if self.control_pso_fitness == 'optimize_in_training':
                    fitness = torch.min(output + self.threshold, torch.ones(1).to(self.device)) + eps
                elif self.control_pso_fitness == 'optimize_out_training':
                    fitness = (1. - torch.min(output + self.threshold, torch.ones(1).to(self.device))) + eps
                else:
                    raise ValueError(self.control_pso_fitness)

        return fitness.detach().cpu().numpy(), img_rescaled, img

    def fitness_pso_inverter(self, dim_space, pos, img, eps=0.1, w_ass=1.0, w_rec=1.0):

        fitness_assessor, img_rescaled, img_rec =  self.fitness(dim_space=dim_space, pos=pos, eps=eps)
        fitness_assessor= w_ass * fitness_assessor

        if isinstance(img, torch.Tensor):
            pass
        else:
            #print(f'Unsupported input type `{dtype}`! Convert to Torch Tensor!')
            img = torch.from_numpy(img)

        img = img.to(self.device)
        img_rec = img_rec.to(self.device)
        fitness_pix_rec =  torch.mean((img.float() - img_rec.float()) ** 2)
        fitness_pix_rec = w_rec * fitness_pix_rec.detach().cpu().numpy()
        fitness = fitness_assessor + fitness_pix_rec

        return fitness + eps, img_rescaled, img_rec

    def load_gan(self):
        pass
    def load_cnn(self):
        pass