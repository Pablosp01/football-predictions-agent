"""Dixon-Coles Poisson goal model.

Each team has attack and defence strengths (log scale, mean-centred).
Expected goals:  xG_home = exp(mu + home_adv + att_home - def_away)
                  xG_away = exp(mu + att_away - def_home)
Scores are independent Poisson, corrected for low-score dependence by
Dixon-Coles rho (adjusts P(0-0), P(1-0), P(0-1), P(1-1)).

Fitted by maximum likelihood (scipy-free: simple gradient-free coordinate
descent on the log-likelihood, which converges well for this size).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.types import Match

MAX_GOALS = 10


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def _dixon_coles_factor(h: int, a: int, rho: float) -> float:
    if h == 0 and a == 0:
        return 1.0 - rho
    if h == 1 and a == 0:
        return 1.0 + rho
    if h == 0 and a == 1:
        return 1.0 + rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class TeamStrength:
    attack: float
    defence: float


class DixonColesModel:
    def __init__(self, rho: float = -0.1, home_advantage: float = 0.25):
        self.rho = rho
        self.home_advantage = home_advantage
        self.mu = 0.0  # log baseline (away goals)
        self.strengths: dict[str, TeamStrength] = {}

    # -- fitting ---------------------------------------------------------
    def fit(self, matches: list[Match], steps: int = 60, lr: float = 0.05) -> "DixonColesModel":
        played = [m for m in matches if m.played]
        if not played:
            raise ValueError("DixonColesModel.fit needs at least one played match")
        teams = sorted({t for m in played for t in (m.home_team, m.away_team)})
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        # params: mu, home_adv, attack[n], defence[n]; centre attack/defence each step
        att = np.zeros(n)
        dfn = np.zeros(n)
        mu = float(np.log(np.mean([m.away_goals for m in played])))  # type: ignore[arg-type]
        ha = 0.25
        hg = np.array([m.home_goals for m in played], dtype=float)
        ag = np.array([m.away_goals for m in played], dtype=float)
        hi = np.array([idx[m.home_team] for m in played])
        ai = np.array([idx[m.away_team] for m in played])
        for _ in range(steps):
            lam_h = np.exp(mu + ha + att[hi] - dfn[ai])
            lam_a = np.exp(mu + att[ai] - dfn[hi])
            # Gradient of the Poisson log-likelihood wrt log-lambda, then
            # chain rule. bincount sums residuals per team (no empty slices).
            gh = hg - lam_h
            ga = ag - lam_a
            mu += lr * float(np.mean(gh + ga))
            ha += lr * float(np.mean(gh))
            cnt_h = np.bincount(hi, minlength=n).astype(float)
            cnt_a = np.bincount(ai, minlength=n).astype(float)
            grad_att = (np.bincount(hi, weights=gh, minlength=n)
                        + np.bincount(ai, weights=ga, minlength=n)) / np.maximum(cnt_h + cnt_a, 1)
            grad_def = -(np.bincount(hi, weights=ga, minlength=n)
                         + np.bincount(ai, weights=gh, minlength=n)) / np.maximum(cnt_h + cnt_a, 1)
            att += lr * grad_att
            dfn += lr * grad_def
            att -= att.mean()
            dfn -= dfn.mean()
        self.mu = mu
        self.home_advantage = ha
        self.strengths = {t: TeamStrength(float(att[idx[t]]), float(dfn[idx[t]])) for t in teams}
        return self

    def _strength(self, team: str) -> TeamStrength:
        return self.strengths.get(team, TeamStrength(0.0, 0.0))

    # -- prediction ------------------------------------------------------
    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        hs, as_ = self._strength(home), self._strength(away)
        xh = math.exp(self.mu + self.home_advantage + hs.attack - as_.defence)
        xa = math.exp(self.mu + as_.attack - hs.defence)
        return xh, xa

    def score_matrix(self, xh: float, xa: float) -> np.ndarray:
        m = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                m[h, a] = (poisson_pmf(h, xh) * poisson_pmf(a, xa)
                           * _dixon_coles_factor(h, a, self.rho))
        m /= m.sum()  # renormalise after rho correction
        return m

    def outcome_probs(self, home: str, away: str) -> tuple[float, float, float]:
        xh, xa = self.expected_goals(home, away)
        m = self.score_matrix(xh, xa)
        p_home = float(np.triu(m, k=1).sum()) if False else float(sum(m[h, a] for h in range(11) for a in range(11) if h > a))
        p_draw = float(sum(m[i, i] for i in range(11)))
        p_away = float(sum(m[h, a] for h in range(11) for a in range(11) if h < a))
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total
