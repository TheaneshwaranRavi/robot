"""AMCL-like particle filter: odometry motion model + likelihood-field scan model (numpy only)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .nav_grid import Grid


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def relative_motion(p0, p1) -> tuple[float, float, float]:
    """Pose p1 expressed in the frame of p0 (both (x, y, yaw) in the same frame)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    c, s = math.cos(p0[2]), math.sin(p0[2])
    return c * dx + s * dy, -s * dx + c * dy, float(wrap(p1[2] - p0[2]))


def compose(p, d) -> tuple[float, float, float]:
    """Apply the robot-frame displacement d to pose p."""
    c, s = math.cos(p[2]), math.sin(p[2])
    return p[0] + c * d[0] - s * d[1], p[1] + s * d[0] + c * d[1], float(wrap(p[2] + d[2]))


@dataclass
class PFConfig:
    n_particles: int = 500
    beams: int = 60
    sigma_hit: float = 0.10       # m, likelihood-field std
    z_hit: float = 0.9
    z_rand: float = 0.1
    max_field: float = 1.0        # m, distance cap of the likelihood field
    beam_weight: float = 0.5      # exponent on the joint beam likelihood (beams are correlated)
    # odometry noise (AMCL "diff" model style), std per unit motion
    a_trans_trans: float = 0.10
    a_trans_rot: float = 0.02
    a_rot_rot: float = 0.10
    a_rot_trans: float = 0.08
    init_xy_std: float = 0.05
    init_yaw_std: float = 0.03
    update_min_d: float = 0.01    # only weight scans after this much motion (AMCL behaviour)
    update_min_a: float = 0.01
    resample_neff: float = 0.5


class LikelihoodField:
    """Likelihood-field beam model on a precomputed signed distance field.

    The signed distance (m) to the nearest obstacle surface is stored at cell centres (positive in
    free space, negative inside obstacles) and bilinearly interpolated at beam endpoints, so the
    model is smooth below the 5 cm cell size and penalises endpoints that land inside obstacles
    as much as endpoints that fall short (an unsigned, nearest-cell field biases the estimate by
    up to a cell towards walls).
    """

    def __init__(self, grid: Grid, dist: np.ndarray | None, cfg: PFConfig, range_max: float) -> None:
        from .nav_grid import distance_field
        self.cfg = cfg
        half = grid.res / 2.0
        out = distance_field(grid.occ, grid.res, cfg.max_field + grid.res) if dist is None else dist
        inside = distance_field(~grid.occ, grid.res, cfg.max_field + grid.res)
        # cell centres adjacent across a surface are +-res/2 from it
        self.sd = np.where(grid.occ, -(inside - half), out - half).astype(np.float64)
        self.inv2s2 = 0.5 / cfg.sigma_hit ** 2
        self.z_hit = cfg.z_hit
        self.z_rand = cfg.z_rand / range_max
        self.res = grid.res
        self.ox, self.oy = grid.origin
        self.ny, self.nx = grid.occ.shape

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated |signed distance| (m) at map points, capped at max_field."""
        fx = (x - self.ox) / self.res - 0.5
        fy = (y - self.oy) / self.res - 0.5
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        tx, ty = fx - ix, fy - iy
        ok = (ix >= 0) & (ix < self.nx - 1) & (iy >= 0) & (iy < self.ny - 1)
        ix = np.where(ok, ix, 0)
        iy = np.where(ok, iy, 0)
        sd = self.sd
        d = ((sd[iy, ix] * (1 - tx) + sd[iy, ix + 1] * tx) * (1 - ty)
             + (sd[iy + 1, ix] * (1 - tx) + sd[iy + 1, ix + 1] * tx) * ty)
        d = np.minimum(np.abs(d), self.cfg.max_field)
        return np.where(ok, d, self.cfg.max_field)

    def lookup(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log-likelihood of beam endpoints at map points."""
        d = self.distance(x, y)
        return np.log(self.z_hit * np.exp(-self.inv2s2 * d * d) + self.z_rand)


class ParticleFilter:
    def __init__(self, grid: Grid, dist: np.ndarray, range_max: float, cfg: PFConfig | None = None,
                 seed: int | None = None) -> None:
        self.cfg = cfg or PFConfig()
        self.field = LikelihoodField(grid, dist, self.cfg, range_max)
        self.range_max = range_max
        self.rng = np.random.default_rng(seed)
        self.particles = np.zeros((self.cfg.n_particles, 3))
        self.weights = np.full(self.cfg.n_particles, 1.0 / self.cfg.n_particles)
        self._moved = np.zeros(2)  # accumulated |trans|, |rot| since the last scan weighting

    def init(self, pose, xy_std: float | None = None, yaw_std: float | None = None) -> None:
        c, n = self.cfg, self.cfg.n_particles
        xy_std = c.init_xy_std if xy_std is None else xy_std
        yaw_std = c.init_yaw_std if yaw_std is None else yaw_std
        self.particles = np.empty((n, 3))
        self.particles[:, 0] = pose[0] + self.rng.normal(0, xy_std, n)
        self.particles[:, 1] = pose[1] + self.rng.normal(0, xy_std, n)
        self.particles[:, 2] = wrap(pose[2] + self.rng.normal(0, yaw_std, n))
        self.weights = np.full(n, 1.0 / n)
        self._moved[:] = 0.0
        self._moved[0] = c.update_min_d  # allow the first scan to be weighted

    def predict(self, d) -> None:
        """Motion update with the odometry displacement d = (dx, dy, dyaw) in the robot frame."""
        dx, dy, dth = d
        trans = math.hypot(dx, dy)
        rot = abs(dth)
        if trans < 1e-6 and rot < 1e-6:
            return
        c, n = self.cfg, len(self.particles)
        s_t = c.a_trans_trans * trans + c.a_trans_rot * rot
        s_r = c.a_rot_rot * rot + c.a_rot_trans * trans
        ndx = dx + self.rng.normal(0, s_t, n)
        ndy = dy + self.rng.normal(0, 0.5 * s_t, n)
        nth = dth + self.rng.normal(0, s_r, n)
        p = self.particles
        cth, sth = np.cos(p[:, 2]), np.sin(p[:, 2])
        p[:, 0] += cth * ndx - sth * ndy
        p[:, 1] += sth * ndx + cth * ndy
        p[:, 2] = wrap(p[:, 2] + nth)
        self._moved += (trans, rot)

    def correct(self, ranges: np.ndarray, angles: np.ndarray, force: bool = False) -> bool:
        """Weight particles by a scan (robot-frame angles). Returns True if a weighting happened."""
        c = self.cfg
        if not force and self._moved[0] < c.update_min_d and self._moved[1] < c.update_min_a:
            return False
        ranges = np.asarray(ranges, dtype=float)
        idx = np.linspace(0, len(ranges) - 1, min(c.beams, len(ranges))).round().astype(int)
        r, a = ranges[idx], np.asarray(angles)[idx]
        ok = np.isfinite(r) & (r < self.range_max - 1e-3) & (r > 0.05)
        r, a = r[ok], a[ok]
        if len(r) < 5:
            return False
        p = self.particles
        ang = p[:, 2:3] + a[None, :]                       # (N, B)
        ex = p[:, 0:1] + r[None, :] * np.cos(ang)
        ey = p[:, 1:2] + r[None, :] * np.sin(ang)
        loglik = self.field.lookup(ex, ey).sum(axis=1) * c.beam_weight
        logw = np.log(np.maximum(self.weights, 1e-300)) + loglik
        logw -= logw.max()
        w = np.exp(logw)
        self.weights = w / w.sum()
        self._moved[:] = 0.0
        if self.n_eff() < c.resample_neff * len(w):
            self.resample()
        return True

    def n_eff(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    def resample(self) -> None:
        """Low-variance (systematic) resampling."""
        n = len(self.weights)
        positions = (self.rng.random() + np.arange(n)) / n
        cum = np.cumsum(self.weights)
        cum[-1] = 1.0
        idx = np.searchsorted(cum, positions)
        self.particles = self.particles[idx].copy()
        self.weights = np.full(n, 1.0 / n)

    def estimate(self) -> tuple[tuple[float, float, float], float, float]:
        """Weighted mean pose, std_xy (RMS of the x and y std) and circular std of yaw."""
        w, p = self.weights, self.particles
        mx, my = float(w @ p[:, 0]), float(w @ p[:, 1])
        sc, ss = float(w @ np.cos(p[:, 2])), float(w @ np.sin(p[:, 2]))
        yaw = math.atan2(ss, sc)
        var = float(w @ ((p[:, 0] - mx) ** 2 + (p[:, 1] - my) ** 2)) / 2.0
        r = min(1.0, math.hypot(sc, ss))
        std_yaw = math.sqrt(max(0.0, -2.0 * math.log(max(r, 1e-12))))
        return (mx, my, yaw), math.sqrt(var), std_yaw
