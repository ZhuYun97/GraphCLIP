import torch
from torch_geometric.utils import dropout_edge
import  copy


def adversarial_aug_train(model_graph, model_text, node_attack, perturb_shapes, step_size, m):
    model_graph.train()
    model_text.train()
    perturbs = []
    for perturb_shape in perturb_shapes:
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size)
        perturb.requires_grad_()
        perturbs.append(perturb)
    
    loss = node_attack(perturbs)
    loss /= m

    for i in range(m-1):
        loss.backward()
        for perturb in perturbs:
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

        loss = node_attack(perturbs)
        loss /=  m

    return loss

def graph_aug(g, f_p, e_p):
    new_g = copy.deepcopy(g)
    drop_mask = torch.empty(
        (g.x.size(1), ),
        dtype=torch.float32,
        device=g.x.device).uniform_(0, 1) < f_p
    
    new_g.x[:, drop_mask] = 0
    e, _ = dropout_edge(new_g.edge_index, p=e_p)
    new_g.edge_index = e
    return new_g