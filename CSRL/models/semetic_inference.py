import torch
import torch.nn as nn

class ConditionalGnn(torch.nn.Module):
    def __init__(self, emb_dim, fp_dim, semetic_dim, num_domain, num_class):
        super(ConditionalGnn, self).__init__()
        self.emb_dim = emb_dim
        self.class_emb = torch.nn.Parameter(
            torch.zeros(num_domain, emb_dim)
        )
        self.backend = nn.Linear(fp_dim, semetic_dim)
        self.predictor = nn.Linear(semetic_dim + emb_dim, num_class)

    def forward(self, batched_data, domains):
        domain_feat = torch.index_select(self.class_emb, 0, domains)
        graph_feat = self.backend(batched_data)
        result = self.predictor(torch.cat([graph_feat, domain_feat], dim=1))
        return result


class DomainClassifier(torch.nn.Module):
    def __init__(self, backend_dim, fp_dim, semetic_dim, num_domain, num_task):
        super(DomainClassifier, self).__init__()
        self.backend = nn.Linear(fp_dim, semetic_dim)
        self.num_task = num_task
        self.predictor = nn.Linear(backend_dim + num_task, num_domain)

    def forward(self, batched_data):
        graph_feat = self.backend(batched_data['fp'])
        y_part = torch.nan_to_num(batched_data['y']).float()
        y_part = y_part.reshape(len(y_part), self.num_task)
        return self.predictor(torch.cat([graph_feat, y_part], dim=-1))


from torch.autograd import Function
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_domain):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_domain)  # 假设有n个域
        )
        self.GRL=GRL()

    def forward(self, x, alpha):
        x = self.GRL.apply(x, alpha)
        domina_pred = self.network(x)
        return domina_pred
