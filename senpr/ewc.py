class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.fisher = {}
        self.params = {}

        self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        for n, p in self.model.named_parameters():
            self.fisher[n] = torch.zeros_like(p)
            self.params[n] = p.clone().detach()

        self.model.eval()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            loss = F.binary_cross_entropy_with_logits(
                self.model(x), y
            )
            loss.backward()

            for n, p in self.model.named_parameters():
                self.fisher[n] += p.grad ** 2

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.params[n])**2).sum()
        return loss
