from __future__ import annotations

import torch
from torch import nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood(emissions, tags, mask)

    def log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        return numerator - denominator

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return -self.log_likelihood(emissions, tags, mask).mean()

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[torch.arange(batch_size), 0, tags[:, 0]]

        for i in range(1, seq_len):
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]
            transition_score = self.transitions[prev_tags, curr_tags]
            emission_score = emissions[torch.arange(batch_size), i, curr_tags]
            score += (transition_score + emission_score) * mask[:, i]

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size), lengths]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        mask = mask.bool()
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history: list[torch.Tensor] = []

        for i in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            best_score, best_path = next_score.max(dim=1)
            history.append(best_path)
            score = torch.where(mask[:, i].unsqueeze(1), best_score, score)

        score += self.end_transitions
        best_last_score, best_last_tag = score.max(dim=1)

        paths: list[list[int]] = []
        lengths = mask.long().sum(dim=1).tolist()
        for batch_idx in range(batch_size):
            seq_len_i = lengths[batch_idx]
            tag = int(best_last_tag[batch_idx].item())
            path = [tag]
            for hist in reversed(history[: seq_len_i - 1]):
                tag = int(hist[batch_idx, tag].item())
                path.append(tag)
            paths.append(list(reversed(path)))
        return paths
