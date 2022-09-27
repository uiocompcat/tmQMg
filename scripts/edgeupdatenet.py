import torch
from torch import nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from qsar_flash.backbones.nn import RadialBasis


class EdgeUpdateBlock(nn.Module):
    def __init__(self, in_features, C):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 2 * C),
            ShiftedSoftplus(),
            nn.Linear(2 * C, C),
        )

    def forward(self, h, edge_attr, edge_index):
        h1 = h[edge_index[0, :]]
        h2 = h[edge_index[1, :]]
        concat = torch.cat((h1, h2, edge_attr), dim=1)
        x = self.mlp(concat)
        return x

    def reset_parameters(self):
        # glorot_orthogonal(self.mlp[0].weight, scale=2.0)
        self.mlp[0].bias.data.fill_(0)  # type: ignore
        # glorot_orthogonal(self.mlp[2].weight, scale=2.0)
        self.mlp[2].bias.data.fill_(0)  # type: ignore


class InteractionBlock(MessagePassing):
    def __init__(self, C):
        super().__init__()
        self.fc1 = nn.Linear(C, C)
        self.mlp = nn.Sequential(
            nn.Linear(C, C),
            ShiftedSoftplus(),
            nn.Linear(C, C),
            ShiftedSoftplus(),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(C, C),
            ShiftedSoftplus(),
            nn.Linear(C, C),
        )

    def forward(self, x, edge_attr, edge_index):
        h = self.fc1(x)
        edge = self.mlp(edge_attr)
        m = self.propagate(edge_index, x=h, edge_attr=edge)
        m = self.mlp1(m)

        return x + m

    def message(self, x_j, edge_attr):
        return x_j * edge_attr

    def reset_parameters(self):
        # glorot_orthogonal(self.fc1.weight, scale=2.0)
        self.fc1.bias.data.fill_(0)
        # glorot_orthogonal(self.mlp[0].weight, scale=2.0)
        self.mlp[0].bias.data.fill_(0)  # type: ignore
        # glorot_orthogonal(self.mlp[2].weight, scale=2.0)
        self.mlp[2].bias.data.fill_(0)  # type: ignore
        # glorot_orthogonal(self.mlp1[0].weight, scale=2.0)
        self.mlp1[0].bias.data.fill_(0)  # type: ignore
        # glorot_orthogonal(self.mlp1[2].weight, scale=2.0)
        self.mlp1[2].bias.data.fill_(0)  # type: ignore


class EdgeUpdateNetBackbone(nn.Module):
    def __init__(
        self,
        C=64,
        num_interactions=3,
        num_gaussians=128,
        cutoff=10.0,
        out_channels=1,
    ):
        super().__init__()

        self.cutoff = cutoff

        self.embedding = nn.Embedding(95, C)
        self.distance_expansion = RadialBasis(
            num_gaussians,
            cutoff=cutoff,
            rbf={"name": "gaussian"},
            envelope={"name": "polynomial", "exponent": 5},
        )

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(C)
            self.interactions.append(block)

        self.edge_updates = nn.ModuleList()
        block = EdgeUpdateBlock(num_gaussians + 2 * C, C)
        self.edge_updates.append(block)

        for _ in range(num_interactions - 1):
            block = EdgeUpdateBlock(C * 3, C)
            self.edge_updates.append(block)

        self.mlp = nn.Sequential(
            nn.Linear(C, C // 2),
            ShiftedSoftplus(),
            nn.Linear(C // 2, out_channels),
        )

        self.reset_parameters()

    def forward(self, z, pos, batch=None):
        assert z.dtype == torch.long

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        distances = (pos[row] - pos[col]).norm(dim=-1)

        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_attr = self.distance_expansion(distances)

        for edge_update, interaction in zip(self.edge_updates, self.interactions):
            edge_attr = edge_update(h, edge_attr, edge_index)
            h = interaction(h, edge_attr, edge_index)
        h = self.mlp(h)

        return h

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()  # type: ignore
        for edge_update in self.edge_updates:
            edge_update.reset_parameters()  # type: ignore
        # glorot_orthogonal(self.mlp[0].weight, scale=2.0)
        self.mlp[0].bias.data.fill_(0)  # type: ignore
        self.mlp[2].weight.data.fill_(0)  # type: ignore
        self.mlp[2].bias.data.fill_(0)  # type: ignore
