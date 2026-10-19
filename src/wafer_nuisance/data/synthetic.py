"""Synthetic die images for die-to-die inspection.

Physical model (all lengths in pixels):

    reflectance r(u, v) --defects--> r'(u, v) --PSF blur--> gain * . + offset --> + noise = image

* A ``DieDesign`` is the layout shared by every die on the wafer: a guillotine partition of the die
  into rectangular blocks, each a line/space array, a contact-hole array, or blank field.
* Each die gets an independent ``NuisanceDraw``: sub-pixel misalignment, CD (linewidth) bias,
  line-edge roughness, focus (PSF width), illumination gain/offset and sensor noise. These are the
  only sources of variation between defect-free dies, so they *define* the normal distribution P
  at every die position. Dies are i.i.d. given the design, which is what makes "one candidate
  from a fresh die" an i.i.d. sample. (Real wafers violate this with radial process signatures;
  that is deliberately left out.)
* Defects edit the reflectance map before imaging, so they are blurred and noised like the pattern.

Coordinates: pixel (row i, col j) has image coordinates (x, y) = (j + 0.5, i + 0.5). Die
coordinates are (u, v) = (x + dx, y + dy), where (dx, dy) is the die's misalignment. The design and
all defect geometry live in die coordinates, so misalignment moves pattern and defects together.

Ground-truth mask: pixels where the defect footprint covers at least half the pixel *and* the
defect changes the surface there (a bridge counts only over the space it fills, not the lines it
overlaps). The mask depends on defect size and shape, never on contrast, so contrast sweeps share
masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom as ndimage_zoom
from scipy.special import expit

from wafer_nuisance.utils.seeding import item_rng

DIE_SIZE = 256  # px, default die side (layout designs and dies)

# Width (px) of the logistic edge used to anti-alias every boundary before PSF blur.
EDGE_WIDTH = 0.3

BlockKind = Literal["lines", "contacts", "blank"]
DefectKind = Literal["particle", "scratch", "bridge", "open", "missing_contact"]
DEFECT_KINDS: tuple[DefectKind, ...] = ("particle", "scratch", "bridge", "open", "missing_contact")


# --------------------------------------------------------------------------------------------
# Design (shared by all dies)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """Axis-aligned rectangle [x0, x1) x [y0, y1) in die coordinates holding one pattern.

    lines:    ``pitch`` = period across the lines, ``width`` = nominal linewidth,
              ``orientation`` "v" means vertical lines (periodic in u), "h" horizontal.
              Line centres sit at phase + pitch/2 + k*pitch along the across-axis.
    contacts: square lattice with period ``pitch`` on both axes, ``width`` = nominal radius.
              Contact centres sit at (phase_u + pitch/2 + i*pitch, phase_v + pitch/2 + j*pitch).
    blank:    constant reflectance ``level``.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    kind: BlockKind
    pitch: float = 0.0
    width: float = 0.0
    phase_u: float = 0.0
    phase_v: float = 0.0
    orientation: Literal["v", "h"] = "v"
    level: float = 0.0

    def __post_init__(self) -> None:
        if not (self.x1 > self.x0 and self.y1 > self.y0):
            raise ValueError(f"empty block {self}")
        if self.kind == "lines" and not 0 < self.width < self.pitch:
            raise ValueError("lines need 0 < width < pitch")
        if self.kind == "contacts" and not 0 < 2 * self.width < self.pitch:
            raise ValueError("contacts need 0 < 2*radius < pitch")

    def across_along(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """For a lines block: (across coordinate, along coordinate, across phase)."""
        if self.orientation == "v":
            return u, v, self.phase_u
        return v, u, self.phase_v

    def across_range(self) -> tuple[int, int]:
        return (self.x0, self.x1) if self.orientation == "v" else (self.y0, self.y1)

    def along_range(self) -> tuple[int, int]:
        return (self.y0, self.y1) if self.orientation == "v" else (self.x0, self.x1)


@dataclass(frozen=True)
class DieDesign:
    height: int
    width: int
    blocks: tuple[Block, ...]

    def blocks_of_kind(self, kind: BlockKind) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if b.kind == kind)

    def block_index_map(self) -> np.ndarray:
        """(H, W) int array giving the block index of each pixel at zero misalignment."""
        out = np.full((self.height, self.width), -1, dtype=np.int64)
        for k, b in enumerate(self.blocks):
            out[b.y0 : b.y1, b.x0 : b.x1] = k
        return out


def _random_lines_block(rng: np.random.Generator, x0: int, y0: int, x1: int, y1: int) -> Block:
    pitch = float(rng.choice([8.0, 10.0, 12.0, 16.0]))
    return Block(
        x0, y0, x1, y1, "lines",
        pitch=pitch,
        width=pitch * rng.uniform(0.4, 0.6),
        phase_u=rng.uniform(0, pitch),
        phase_v=rng.uniform(0, pitch),
        orientation="v" if rng.random() < 0.5 else "h",
    )


def _random_contacts_block(rng: np.random.Generator, x0: int, y0: int, x1: int, y1: int) -> Block:
    pitch = float(rng.choice([10.0, 12.0, 14.0, 16.0]))
    return Block(
        x0, y0, x1, y1, "contacts",
        pitch=pitch,
        width=pitch * rng.uniform(0.25, 0.35),
        phase_u=rng.uniform(0, pitch),
        phase_v=rng.uniform(0, pitch),
    )


def make_layout_design(
    rng: np.random.Generator, size: int = DIE_SIZE, min_block: int = 48, max_depth: int = 3
) -> DieDesign:
    """Random guillotine layout. Always contains at least one lines and one contacts block, so
    every defect kind can be placed."""
    rects: list[tuple[int, int, int, int]] = []

    def split(x0: int, y0: int, x1: int, y1: int, depth: int) -> None:
        w, h = x1 - x0, y1 - y0
        can_split = max(w, h) >= 2 * min_block
        if depth >= max_depth or not can_split or (depth > 0 and rng.random() < 0.2):
            rects.append((x0, y0, x1, y1))
            return
        if w >= h:
            s = int(rng.integers(x0 + min_block, x1 - min_block + 1))
            split(x0, y0, s, y1, depth + 1)
            split(s, y0, x1, y1, depth + 1)
        else:
            s = int(rng.integers(y0 + min_block, y1 - min_block + 1))
            split(x0, y0, x1, s, depth + 1)
            split(x0, s, x1, y1, depth + 1)

    split(0, 0, size, size, 0)
    if len(rects) < 2:
        raise ValueError("layout needs at least two blocks; increase size or max_depth")

    kinds: list[BlockKind] = list(rng.choice(["lines", "contacts", "blank"], size=len(rects),
                                             p=[0.5, 0.3, 0.2]))
    # Guarantee one lines and one contacts block by overwriting the largest eligible blocks.
    by_area = sorted(range(len(rects)),
                     key=lambda k: -(rects[k][2] - rects[k][0]) * (rects[k][3] - rects[k][1]))
    if "lines" not in kinds:
        kinds[by_area[0]] = "lines"
    if "contacts" not in kinds:
        only_lines = kinds.index("lines") if kinds.count("lines") == 1 else None
        kinds[next(k for k in by_area if k != only_lines)] = "contacts"

    blocks = []
    for (x0, y0, x1, y1), kind in zip(rects, kinds, strict=True):
        if kind == "lines":
            blocks.append(_random_lines_block(rng, x0, y0, x1, y1))
        elif kind == "contacts":
            blocks.append(_random_contacts_block(rng, x0, y0, x1, y1))
        else:
            blocks.append(Block(x0, y0, x1, y1, "blank", level=float(rng.choice([0.0, 1.0]))))
    return DieDesign(size, size, tuple(blocks))


def make_uniform_design(
    kind: Literal["lines", "contacts"],
    size: int = DIE_SIZE,
    pitch: float = 12.0,
    width: float | None = None,
    orientation: Literal["v", "h"] = "v",
) -> DieDesign:
    """A single stationary pattern covering the whole die."""
    if width is None:
        width = 0.5 * pitch if kind == "lines" else 0.3 * pitch
    block = Block(0, 0, size, size, kind, pitch=pitch, width=width, orientation=orientation)
    return DieDesign(size, size, (block,))


# --------------------------------------------------------------------------------------------
# Nuisance (per-die variation that is NOT a defect)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NuisanceConfig:
    """Distribution of per-die nuisance. Defaults are moderate; set any std to 0 to remove it.

    A nonzero ``cd_bias_mean`` or shifted ``psf_sigma`` models systematic process/tool drift.

    The four nuisance sources a die-to-die difference image picks up, and their strength knobs:

    * misalignment residue at pattern edges: ``warp_std``, a smooth local displacement that a
      global (rigid) alignment cannot remove; ``shift_std`` is the global part, which alignment
      removes up to its own sub-pixel error;
    * per-die linewidth variation: ``cd_bias_std``;
    * local intensity blobs (e.g. film-thickness or illumination patches): ``blob_rate``
      (expected count per die), ``blob_amp_min``/``blob_amp_max`` (|amplitude|, image units) and
      ``blob_sigma_min``/``blob_sigma_max`` (px); ``blob_min_gap`` keeps blobs apart (centres at
      least ``blob_min_gap * (sigma_i + sigma_j)`` px apart), so overlapping blobs cannot add up
      and a blob's peak in the difference image is bounded by about ``blob_amp_max``;
    * line-edge roughness: ``ler_std``.

    The warp and blobs default to off; the regimes in ``data.config`` turn them on.
    """

    shift_std: float = 0.3  # px, misalignment per axis
    cd_bias_mean: float = 0.0  # px, systematic linewidth offset (drift)
    cd_bias_std: float = 0.15  # px, per-die linewidth variation
    ler_std: float = 0.25  # px, line-edge roughness amplitude
    ler_corr_len: float = 2.0  # px, roughness correlation length
    psf_sigma: float = 1.0  # px, mean blur (focus)
    psf_sigma_std: float = 0.05  # px
    gain_mean: float = 0.6
    gain_std: float = 0.02
    offset_mean: float = 0.2
    offset_std: float = 0.01
    noise_std: float = 0.02  # additive Gaussian sensor noise, image units
    warp_std: float = 0.0  # px, std of the local displacement at each control point
    warp_grid: int = 8  # control grid is (warp_grid + 1)^2 points spread over the die
    blob_rate: float = 0.0  # expected number of intensity blobs per die (Poisson)
    blob_amp_min: float = 0.0  # image units
    blob_amp_max: float = 0.0  # image units
    blob_sigma_min: float = 1.5  # px
    blob_sigma_max: float = 4.0  # px
    blob_min_gap: float = 0.0  # min centre distance / (sigma_i + sigma_j); 0 = independent
    blob_max_tries: int = 50  # redraws of a blob that is too close to an earlier one, then drop


@dataclass(frozen=True)
class Blob:
    """Gaussian intensity bump added to the image: amp * exp(-r^2 / (2 sigma^2)). The centre
    (fy, fx) is a fraction of the die height/width."""

    fy: float
    fx: float
    sigma: float
    amp: float


@dataclass(frozen=True)
class NuisanceDraw:
    """One die's realised nuisance. Rendering is a deterministic function of (design, draw,
    defects), so a die can be re-rendered with and without defects under identical nuisance."""

    dx: float
    dy: float
    cd_bias: float
    psf_sigma: float
    gain: float
    offset: float
    ler_std: float
    ler_corr_len: float
    noise_std: float
    ler_seed: int
    noise_seed: int
    warp_std: float = 0.0
    warp_grid: int = 8
    warp_seed: int = 0
    blobs: tuple[Blob, ...] = ()


def draw_nuisance(cfg: NuisanceConfig, rng: np.random.Generator) -> NuisanceDraw:
    # The original fields are drawn first and in their original order, so adding the warp and
    # blobs did not change the draws of the original fields.
    base = dict(
        dx=rng.normal(0.0, cfg.shift_std),
        dy=rng.normal(0.0, cfg.shift_std),
        cd_bias=rng.normal(cfg.cd_bias_mean, cfg.cd_bias_std),
        psf_sigma=max(0.0, rng.normal(cfg.psf_sigma, cfg.psf_sigma_std)),
        gain=rng.normal(cfg.gain_mean, cfg.gain_std),
        offset=rng.normal(cfg.offset_mean, cfg.offset_std),
        ler_std=cfg.ler_std,
        ler_corr_len=cfg.ler_corr_len,
        noise_std=cfg.noise_std,
        ler_seed=int(rng.integers(2**63 - 1)),
        noise_seed=int(rng.integers(2**63 - 1)),
    )
    warp_seed = int(rng.integers(2**63 - 1))
    n_blobs = int(rng.poisson(cfg.blob_rate)) if cfg.blob_rate > 0 else 0
    return NuisanceDraw(**base, warp_std=cfg.warp_std, warp_grid=cfg.warp_grid,
                        warp_seed=warp_seed, blobs=_draw_blobs(cfg, rng, n_blobs))


def _draw_blobs(cfg: NuisanceConfig, rng: np.random.Generator, n: int) -> tuple[Blob, ...]:
    """n blobs drawn one after another; a blob closer than ``blob_min_gap * (sigma_i +
    sigma_j)`` px to an earlier one is redrawn (up to ``blob_max_tries`` times, then dropped).
    Distances are measured in pixels of a ``DIE_SIZE``-pixel square die."""
    blobs: list[Blob] = []
    for _ in range(n):
        for _ in range(max(1, cfg.blob_max_tries)):
            b = Blob(fy=rng.uniform(), fx=rng.uniform(),
                     sigma=rng.uniform(cfg.blob_sigma_min, cfg.blob_sigma_max),
                     amp=rng.choice([-1.0, 1.0]) * rng.uniform(cfg.blob_amp_min, cfg.blob_amp_max))
            if all(DIE_SIZE * np.hypot(b.fy - o.fy, b.fx - o.fx)
                   >= cfg.blob_min_gap * (b.sigma + o.sigma) for o in blobs):
                blobs.append(b)
                break
    return tuple(blobs)


# --------------------------------------------------------------------------------------------
# Defects
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DefectSpec:
    """What to inject; *where* is chosen by ``place_defect``.

    ``size`` meaning per kind:
        particle         radius (px)
        scratch          width (px); ``length`` sets its length
        bridge, open     extent along the line direction (px)
        missing_contact  number of adjacent contacts removed (integer >= 1)

    ``contrast`` meaning per kind:
        particle, scratch               additive reflectance change, signed (pattern contrast is 1)
        bridge, open, missing_contact   fill fraction in [0, 1]: 1 = fully shorted / cut / missing
    """

    kind: DefectKind
    size: float
    contrast: float = 1.0
    length: float = 40.0  # scratch only
    aspect: float = 1.0  # particle only: ratio of major to minor axis (>= 1)
    irregularity: float = 0.0  # particle only: 0 = ellipse, ~0.3 = lumpy blob

    def __post_init__(self) -> None:
        if self.kind not in DEFECT_KINDS:
            raise ValueError(f"unknown defect kind {self.kind!r}")
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.kind == "missing_contact" and float(self.size) != int(self.size):
            raise ValueError("missing_contact size is a contact count")
        if self.kind in ("bridge", "open", "missing_contact") and not 0 <= self.contrast <= 1:
            raise ValueError(f"{self.kind} contrast is a fill fraction in [0, 1]")
        if self.aspect < 1:
            raise ValueError("aspect must be >= 1")


class PlacedDefect(Protocol):
    """A defect with concrete geometry in die coordinates."""

    contrast: float

    @property
    def target(self) -> float | None:
        """Reflectance the defect pulls the surface toward, or None for an additive defect."""

    def footprint(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Soft coverage in [0, 1] of each pixel."""


def _soft(sd: np.ndarray) -> np.ndarray:
    """Coverage from a signed distance (positive inside)."""
    return expit(sd / EDGE_WIDTH)


def _soft_box(u, v, cu, cv, half_u, half_v) -> np.ndarray:
    return _soft(np.minimum(half_u - np.abs(u - cu), half_v - np.abs(v - cv)))


@dataclass(frozen=True)
class Particle:
    cu: float
    cv: float
    radius: float
    aspect: float
    angle: float
    harmonics: tuple[tuple[float, float], ...]  # (amplitude, phase) for k = 2, 3, ...
    contrast: float

    @property
    def target(self) -> None:
        return None

    def footprint(self, u, v):
        du, dv = u - self.cu, v - self.cv
        c, s = np.cos(self.angle), np.sin(self.angle)
        # Rotate into the particle frame and squash the minor axis so the major axis = radius.
        a = du * c + dv * s
        b = (-du * s + dv * c) * self.aspect
        rho = np.hypot(a, b)
        theta = np.arctan2(b, a)
        r_theta = np.full_like(rho, self.radius)
        for k, (amp, ph) in enumerate(self.harmonics, start=2):
            r_theta = r_theta + self.radius * amp * np.cos(k * theta + ph)
        return _soft(r_theta - rho)


@dataclass(frozen=True)
class Scratch:
    cu: float
    cv: float
    angle: float
    length: float
    width: float
    contrast: float

    @property
    def target(self) -> None:
        return None

    def footprint(self, u, v):
        du, dv = u - self.cu, v - self.cv
        c, s = np.cos(self.angle), np.sin(self.angle)
        t = np.clip(du * c + dv * s, -self.length / 2, self.length / 2)
        dist = np.hypot(du - t * c, dv - t * s)
        return _soft(self.width / 2 - dist)


@dataclass(frozen=True)
class RectEdit:
    """Bridge (target 1: material fills a space) or open (target 0: a line is cut)."""

    cu: float
    cv: float
    half_u: float
    half_v: float
    fill_to: float
    contrast: float

    @property
    def target(self) -> float:
        return self.fill_to

    def footprint(self, u, v):
        return _soft_box(u, v, self.cu, self.cv, self.half_u, self.half_v)


@dataclass(frozen=True)
class MissingContacts:
    centers: tuple[tuple[float, float], ...]
    radius: float
    contrast: float

    @property
    def target(self) -> float:
        return 0.0

    def footprint(self, u, v):
        out = np.zeros_like(u)
        for cu, cv in self.centers:
            out = np.maximum(out, _soft(self.radius - np.hypot(u - cu, v - cv)))
        return out


# Extra coverage (px) beyond the nominal geometry so per-die CD bias / roughness can't leave a
# sliver of the original pattern inside a bridge, open or missing contact.
_OVERLAP = 1.0


def _pick_lines_site(
    rng: np.random.Generator, blocks: tuple[Block, ...], along_extent: float, on_line: bool
) -> tuple[Block, float, float]:
    """Choose (block, across centre, along centre) for a bridge (on_line=False -> centred on a
    space) or an open (on_line=True -> centred on a line), fully inside its block."""
    candidates = []
    for b in blocks:
        a0, a1 = b.across_range()
        l0, l1 = b.along_range()
        phase = b.phase_u if b.orientation == "v" else b.phase_v
        offset = phase + (b.pitch / 2 if on_line else 0.0)
        ks = np.arange(np.ceil((a0 + b.pitch - offset) / b.pitch),
                       np.floor((a1 - b.pitch - offset) / b.pitch) + 1)
        lo, hi = l0 + along_extent / 2 + 2, l1 - along_extent / 2 - 2
        if ks.size and hi > lo:
            candidates.append((b, offset, ks, lo, hi))
    if not candidates:
        raise ValueError("no lines block large enough for this defect")
    b, offset, ks, lo, hi = candidates[rng.integers(len(candidates))]
    return b, offset + rng.choice(ks) * b.pitch, rng.uniform(lo, hi)


def place_defect(design: DieDesign, spec: DefectSpec, rng: np.random.Generator) -> PlacedDefect:
    """Choose a concrete location/orientation for ``spec`` on ``design``."""
    H, W = design.height, design.width

    if spec.kind == "particle":
        m = spec.size + 2
        harmonics = tuple(
            (spec.irregularity * rng.uniform(-1, 1) / k, rng.uniform(0, 2 * np.pi))
            for k in (2, 3, 4)
        )
        return Particle(rng.uniform(m, W - m), rng.uniform(m, H - m), spec.size, spec.aspect,
                        rng.uniform(0, np.pi), harmonics, spec.contrast)

    if spec.kind == "scratch":
        m = spec.length / 2 + spec.size + 2
        if 2 * m >= min(H, W):
            raise ValueError("scratch longer than the die")
        return Scratch(rng.uniform(m, W - m), rng.uniform(m, H - m), rng.uniform(0, np.pi),
                       spec.length, spec.size, spec.contrast)

    if spec.kind in ("bridge", "open"):
        on_line = spec.kind == "open"
        b, across_c, along_c = _pick_lines_site(rng, design.blocks_of_kind("lines"), spec.size,
                                                on_line)
        half_across = (b.width if on_line else b.pitch - b.width) / 2 + _OVERLAP
        half_along = spec.size / 2
        if b.orientation == "v":
            cu, cv, hu, hv = across_c, along_c, half_across, half_along
        else:
            cu, cv, hu, hv = along_c, across_c, half_along, half_across
        return RectEdit(cu, cv, hu, hv, fill_to=0.0 if on_line else 1.0, contrast=spec.contrast)

    # missing_contact: a horizontal run of `size` adjacent contacts.
    count = int(spec.size)
    candidates = []
    for b in design.blocks_of_kind("contacts"):
        cu0 = b.phase_u + b.pitch / 2
        cv0 = b.phase_v + b.pitch / 2
        iu = np.arange(np.ceil((b.x0 + b.pitch - cu0) / b.pitch),
                       np.floor((b.x1 - b.pitch - cu0) / b.pitch) - count + 2)
        iv = np.arange(np.ceil((b.y0 + b.pitch - cv0) / b.pitch),
                       np.floor((b.y1 - b.pitch - cv0) / b.pitch) + 1)
        if iu.size and iv.size:
            candidates.append((b, cu0, cv0, iu, iv))
    if not candidates:
        raise ValueError(f"no contacts block fits {count} adjacent missing contacts")
    b, cu0, cv0, iu, iv = candidates[rng.integers(len(candidates))]
    i, j = rng.choice(iu), rng.choice(iv)
    centers = tuple((cu0 + (i + t) * b.pitch, cv0 + j * b.pitch) for t in range(count))
    return MissingContacts(centers, b.width + _OVERLAP, spec.contrast)


def random_defect_spec(rng: np.random.Generator, kind: DefectKind | None = None) -> DefectSpec:
    """A defect with parameters drawn from broad default ranges (for mixed test sets; sweeps
    should construct ``DefectSpec`` directly)."""
    if kind is None:
        kind = DEFECT_KINDS[rng.integers(len(DEFECT_KINDS))]
    if kind == "particle":
        return DefectSpec("particle", size=rng.uniform(1.5, 6.0),
                          contrast=rng.choice([-1, 1]) * rng.uniform(0.3, 1.0),
                          aspect=rng.uniform(1.0, 2.0), irregularity=rng.uniform(0.0, 0.4))
    if kind == "scratch":
        return DefectSpec("scratch", size=rng.uniform(1.0, 3.0), contrast=-rng.uniform(0.3, 0.8),
                          length=rng.uniform(20, 80))
    if kind in ("bridge", "open"):
        return DefectSpec(kind, size=rng.uniform(2.0, 8.0), contrast=rng.uniform(0.5, 1.0))
    return DefectSpec("missing_contact", size=int(rng.integers(1, 4)),
                      contrast=rng.uniform(0.5, 1.0))


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def die_coordinates(design: DieDesign, draw: NuisanceDraw) -> tuple[np.ndarray, np.ndarray]:
    """Die coordinates (u, v) of every pixel: global shift (dx, dy) plus the local warp."""
    y, x = np.mgrid[0 : design.height, 0 : design.width].astype(np.float64) + 0.5
    wu, wv = _warp_field((design.height, design.width), draw)
    return x + draw.dx + wu, y + draw.dy + wv


def _warp_field(shape: tuple[int, int], draw: NuisanceDraw) -> tuple[np.ndarray, np.ndarray]:
    """Smooth local displacement (u and v components, px). I.i.d. N(0, warp_std^2) values on a
    (warp_grid + 1)^2 control grid spanning the die, interpolated with a cubic spline, so the
    displacement varies on the scale of one control cell (die size / warp_grid). A rigid
    alignment cannot remove it; what is left shows up at pattern edges in the difference image."""
    if draw.warp_std == 0:
        return np.zeros(shape), np.zeros(shape)
    g = draw.warp_grid + 1
    ctrl = draw.warp_std * np.random.default_rng(draw.warp_seed).standard_normal((2, g, g))
    rows, cols = _zoom_matrix(shape[0], g), _zoom_matrix(shape[1], g)
    wu, wv = (rows @ c @ cols.T for c in ctrl)
    return wu, wv


@lru_cache(maxsize=8)
def _zoom_matrix(n_out: int, g: int) -> np.ndarray:
    """(n_out, g) matrix of 1-D cubic-spline zoom from g control points to n_out samples (the
    corners map to the end control points). Spline zoom is linear and separable, so the 2-D
    zoom of a control grid C is rows @ C @ cols.T, the same as ``scipy.ndimage.zoom(C, ...,
    order=3)`` but much cheaper per die."""
    return np.stack([ndimage_zoom(e, n_out / g, order=3) for e in np.eye(g)], axis=1)


def _blob_image(shape: tuple[int, int], blobs: tuple[Blob, ...]) -> np.ndarray:
    out = np.zeros(shape)
    if not blobs:
        return out
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64) + 0.5
    for b in blobs:
        cy, cx = b.fy * shape[0], b.fx * shape[1]
        out += b.amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * b.sigma**2))
    return out


def _roughness_field(shape: tuple[int, int], draw: NuisanceDraw) -> np.ndarray:
    if draw.ler_std == 0:
        return np.zeros(shape)
    white = np.random.default_rng(draw.ler_seed).standard_normal(shape)
    smooth = gaussian_filter(white, draw.ler_corr_len, mode="wrap")
    return draw.ler_std * smooth / smooth.std()


def design_reflectance(
    design: DieDesign, u: np.ndarray, v: np.ndarray, cd_bias: float, roughness: np.ndarray
) -> np.ndarray:
    """Defect-free reflectance in [0, 1]. ``cd_bias`` widens lines by cd_bias and contact
    radii by cd_bias / 2 (so contact diameter also grows by cd_bias).

    Block boundaries are anti-aliased like every other edge: each block's pattern is weighted by
    its soft box coverage, so neighbouring blocks blend over ~EDGE_WIDTH. (A hard switch at the
    boundary is a point-sampled step: a sub-pixel shift then changes the image by more than a
    translation, and the die-to-die difference lit up every block boundary.)"""
    r = np.zeros_like(u)
    m = 4 * EDGE_WIDTH  # coverage is < 1e-5 beyond this distance outside the block
    for b in design.blocks:
        near = (u >= b.x0 - m) & (u < b.x1 + m) & (v >= b.y0 - m) & (v < b.y1 + m)
        if not near.any():
            continue
        ub, vb, rough = u[near], v[near], roughness[near]
        cover = _soft(np.minimum.reduce([ub - b.x0, b.x1 - ub, vb - b.y0, b.y1 - vb]))
        if b.kind == "blank":
            pattern = np.full_like(ub, b.level)
        elif b.kind == "lines":
            across, _, phase = b.across_along(ub, vb)
            dist = np.abs(np.mod(across - phase, b.pitch) - b.pitch / 2)
            pattern = _soft((b.width + cd_bias) / 2 - dist + rough)
        else:
            du = np.mod(ub - b.phase_u, b.pitch) - b.pitch / 2
            dv = np.mod(vb - b.phase_v, b.pitch) - b.pitch / 2
            pattern = _soft(b.width + cd_bias / 2 - np.hypot(du, dv) + rough)
        r[near] += cover * pattern
    return r


@dataclass(frozen=True)
class Die:
    image: np.ndarray  # (H, W) float32
    mask: np.ndarray  # (H, W) bool, ground-truth defect pixels
    draw: NuisanceDraw
    defects: tuple[PlacedDefect, ...] = field(default=())

    @property
    def is_defective(self) -> bool:
        return bool(self.mask.any())


def render_die(
    design: DieDesign,
    draw: NuisanceDraw,
    defects: tuple[PlacedDefect, ...] = (),
    noiseless: bool = False,
) -> Die:
    u, v = die_coordinates(design, draw)
    rough = _roughness_field(u.shape, draw)
    r_clean = design_reflectance(design, u, v, draw.cd_bias, rough)

    r = r_clean.copy()
    mask = np.zeros(u.shape, dtype=bool)
    for d in defects:
        fp = d.footprint(u, v)
        if d.target is None:
            r = r + d.contrast * fp
            changes_surface = np.ones_like(mask)
        else:
            r = r + d.contrast * fp * (d.target - r)
            changes_surface = np.abs(d.target - r_clean) > 0.5
        mask |= (fp >= 0.5) & changes_surface

    blurred = gaussian_filter(r, draw.psf_sigma, mode="reflect") if draw.psf_sigma > 0 else r
    image = draw.offset + draw.gain * blurred + _blob_image(blurred.shape, draw.blobs)
    if not noiseless and draw.noise_std > 0:
        image = image + draw.noise_std * np.random.default_rng(draw.noise_seed).standard_normal(
            image.shape
        )
    return Die(image.astype(np.float32), mask, draw, tuple(defects))


# --------------------------------------------------------------------------------------------
# Wafer: reproducible streams of dies
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticWafer:
    """All dies share ``design``; each die's nuisance (and defect placement) is keyed by
    (seed, stream, index), so die i of a stream never depends on how many dies were requested.

    Use separate streams for disjoint, independent sets, e.g. "reference", "calibration",
    "test". ``defective_die(i, ..., stream=s)`` has exactly the nuisance of ``normal_die(i, s)``,
    so the pair differs only by the injected defects.
    """

    design: DieDesign
    nuisance: NuisanceConfig
    seed: int

    @classmethod
    def from_seed(
        cls,
        seed: int,
        layout: Literal["layout", "lines", "contacts"] = "layout",
        size: int = DIE_SIZE,
        nuisance: NuisanceConfig | None = None,
    ) -> SyntheticWafer:
        if layout == "layout":
            design = make_layout_design(item_rng(seed, "design", 0), size=size)
        else:
            design = make_uniform_design(layout, size=size)
        return cls(design, nuisance or NuisanceConfig(), seed)

    def draw(self, index: int, stream: str) -> NuisanceDraw:
        return draw_nuisance(self.nuisance, item_rng(self.seed, stream, index))

    def normal_die(self, index: int, stream: str = "reference", noiseless: bool = False) -> Die:
        return render_die(self.design, self.draw(index, stream), noiseless=noiseless)

    def normal_images(self, n: int, stream: str = "reference", start: int = 0) -> np.ndarray:
        """(n, H, W) float32 stack of defect-free dies start, ..., start + n - 1."""
        return np.stack([self.normal_die(i, stream).image for i in range(start, start + n)])

    def defective_die(
        self,
        index: int,
        specs: tuple[DefectSpec, ...] | DefectSpec,
        stream: str = "test",
        noiseless: bool = False,
    ) -> Die:
        if isinstance(specs, DefectSpec):
            specs = (specs,)
        rng = item_rng(self.seed, f"{stream}/defects", index)
        placed = tuple(place_defect(self.design, s, rng) for s in specs)
        return render_die(self.design, self.draw(index, stream), placed, noiseless=noiseless)
