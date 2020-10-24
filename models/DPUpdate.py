from opacus import PrivacyEngine
from opacus.utils import stats
from opacus.utils.module_modification import convert_batchnorm_modules
import numpy as np
import torch
from torch import nn


# Update with differential privacy
class LocalDPUpdate(object):
    def __init__(self, args, train_loader=None, clientID=None):
        self.args = args
        self.ldr_train = train_loader
        self.loss_func = nn.CrossEntropyLoss()
        self.client_id = clientID
    def train(self, net):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        virtual_batch_rate = 4
        virtual_batch_size = virtual_batch_rate * self.args.local_bs
        privacy_engine = PrivacyEngine(net, batch_size=self.args.local_bs, sample_size=len(self.ldr_train),
                                       alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                                       noise_multiplier=0.3, max_grad_norm=1.2, secure_rng=self.args.secure_rng)
        privacy_engine.attach(optimizer)


        epoch_loss = []
        for i in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (attributes, labels) in enumerate(self.ldr_train):
                attributes, labels = attributes.to(self.args.device), labels.to(device=self.args.device,
                                                                                dtype=torch.long)
                net.zero_grad()
                log_probs = net(attributes)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # take a real optimizer step after N_VIRTUAL_STEP steps t
                # if ((i + 1) % virtual_batch_rate == 0) or ((i + 1) == len(self.ldr_train)):
                #     optimizer.step()
                #     optimizer.zero_grad()
                # else:
                #     optimizer.virtual_step()  # take a virtual step

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, batch_idx * len(attributes), len(self.ldr_train.dataset),
                           100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

                if batch_idx % 200 == 0:
                    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self.args.delta)
                    print(
                        f"\tClientID: {self.client_id}"
                        f"\tTrain Epoch: {i} \t"
                        f"Loss: {np.mean(batch_loss):.6f} "
                        f"(ε = {epsilon:.2f}, δ = {self.args.delta})"
                    )
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
