#!/usr/bin/env python3
"""
VASP Volumetric File Parser & Periodic Geometry Utilities.

Supports:
- ELFCAR (ISPIN=1 or 2; automatically reads both spin blocks)
- CHGCAR / CHG / PARCHG (ISPIN=1 or 2; skips PAW augmentation blocks, reads total + spin density)
- LOCPOT (ISPIN=1 or 2; electrostatic potential in eV)
- Optional OUTCAR parsing for automatic local magnetic-moment spin alignment (majority/minority)
- Minimum Image Convention (MIC) distance, cluster unwrapping, covalent bond detection, and 3D cubic-spline interpolation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

# Standard covalent radii (Angstrom, Cordero et al. / Pyykko) for automatic bond detection
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Mo": 1.54, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39,
    "W": 1.62, "Pt": 1.36, "Au": 1.36, "Pb": 1.46, "Bi": 1.48,
}


@dataclass
class VaspVolumetricData:
    filepath: str
    title: str
    lattice: np.ndarray        # (3, 3) row vectors in Angstrom
    elements: List[str]
    counts: List[int]
    frac_coords: np.ndarray    # (N, 3) wrapped into [0, 1)
    cart_coords: np.ndarray    # (N, 3) in Angstrom
    atom_labels: List[str]     # e.g., ["Fe1", "S1", "S2", ...]
    atom_elements: List[str]   # e.g., ["Fe", "S", "S", ...]
    grid_shape: Tuple[int, int, int]  # (nx, ny, nz)
    blocks: List[np.ndarray]   # list of (nx, ny, nz) float64 arrays
    field_kind: str            # "elf", "chg", "locpot", "parchg", "unknown"

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def natoms(self) -> int:
        return len(self.atom_labels)


def infer_field_kind(filepath: str) -> str:
    name = os.path.basename(filepath).upper()
    if "ELF" in name:
        return "elf"
    if "LOCPOT" in name:
        return "locpot"
    if "PARCHG" in name:
        return "parchg"
    if "CHGDIFF" in name or "DIFF" in name:
        return "cdd"
    if "CHG" in name:
        return "chg"
    return "unknown"


def _is_grid_header_line(parts: List[str]) -> bool:
    if len(parts) != 3:
        return False
    return all(re.fullmatch(r"\d+", p) is not None for p in parts)


def read_vasp_volumetric(filepath: str, max_blocks: int = 2) -> VaspVolumetricData:
    """
    Parse VASP volumetric files (ELFCAR, CHGCAR, CHG, LOCPOT, PARCHG).
    Handles:
      - Selective dynamics lines if present
      - Direct or Cartesian atomic coordinates
      - Multiple volumetric blocks (ISPIN=2 ELFCAR / CHGCAR / LOCPOT)
      - PAW augmentation occupancies between CHGCAR blocks
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        title = f.readline().strip()
        scale_line = f.readline().strip().split()
        scale = float(scale_line[0])
        lat = np.array([[float(x) for x in f.readline().split()[:3]] for _ in range(3)], dtype=np.float64)
        if scale < 0:
            # Negative scale in VASP means target cell volume
            vol_cur = abs(np.linalg.det(lat))
            lat = lat * ((-scale / vol_cur) ** (1.0 / 3.0))
        else:
            lat = lat * scale

        line5 = f.readline().split()
        # VASP 5 has element symbols; VASP 4 has integer counts directly
        if all(re.fullmatch(r"\d+", tok) for tok in line5):
            counts = [int(x) for x in line5]
            elements = [f"X{i+1}" for i in range(len(counts))]
        else:
            elements = line5
            counts = [int(x) for x in f.readline().split()]

        coord_header = f.readline().strip()
        if coord_header.lower().startswith("s"):  # Selective dynamics
            coord_header = f.readline().strip()
        is_direct = coord_header.lower().startswith("d")

        n_atoms = sum(counts)
        raw_coords = np.array(
            [[float(x) for x in f.readline().split()[:3]] for _ in range(n_atoms)],
            dtype=np.float64,
        )
        if is_direct:
            frac = raw_coords % 1.0
            cart = frac @ lat
        else:
            cart = raw_coords * (scale if scale > 0 else 1.0)
            frac = (cart @ np.linalg.inv(lat)) % 1.0
            cart = frac @ lat

        atom_labels: List[str] = []
        atom_elements: List[str] = []
        for el, cnt in zip(elements, counts):
            for k in range(1, cnt + 1):
                atom_labels.append(f"{el}{k}")
                atom_elements.append(el)

        # Read grid blocks
        blocks: List[np.ndarray] = []
        grid_shape: Optional[Tuple[int, int, int]] = None

        while len(blocks) < max_blocks:
            # Find next grid dimension line (3 integers)
            ng = None
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if _is_grid_header_line(parts):
                    ng = (int(parts[0]), int(parts[1]), int(parts[2]))
                    break
            if ng is None:
                break
            if grid_shape is None:
                grid_shape = ng
            elif ng != grid_shape:
                break

            npts = ng[0] * ng[1] * ng[2]
            vals: List[float] = []
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                # Guard against unexpected text line
                if parts[0].lower().startswith("augmentation"):
                    break
                if _is_grid_header_line(parts) and len(vals) >= npts:
                    break
                for p in parts:
                    vals.append(float(p))
                if len(vals) >= npts:
                    break

            if len(vals) < npts:
                raise ValueError(
                    f"Incomplete volumetric block in {filepath}: expected {npts} values, got {len(vals)}"
                )

            # VASP writes x-fastest, then y, then z -> reshape (nz, ny, nx) and transpose to (nx, ny, nz)
            arr = np.array(vals[:npts], dtype=np.float64).reshape((ng[2], ng[1], ng[0])).transpose(2, 1, 0)
            blocks.append(arr)

    if grid_shape is None or not blocks:
        raise ValueError(f"No volumetric grid data found in {filepath}")

    return VaspVolumetricData(
        filepath=filepath,
        title=title,
        lattice=lat,
        elements=elements,
        counts=counts,
        frac_coords=frac,
        cart_coords=cart,
        atom_labels=atom_labels,
        atom_elements=atom_elements,
        grid_shape=grid_shape,
        blocks=blocks,
        field_kind=infer_field_kind(filepath),
    )


def read_outcar_magnetization(outcar_path: str) -> Optional[np.ndarray]:
    """
    Parse the final 'magnetization (x)' table from VASP OUTCAR.
    Returns an array of shape (N_atoms,) with total local magnetic moments, or None if not found.
    """
    if not os.path.isfile(outcar_path):
        return None
    last_block: Optional[List[float]] = None
    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if "magnetization (x)" in lines[i]:
            # Skip header lines until dashes
            j = i + 1
            while j < len(lines) and "----" not in lines[j]:
                j += 1
            j += 1
            moms: List[float] = []
            while j < len(lines):
                s = lines[j].strip()
                if not s or s.startswith("---"):
                    break
                parts = s.split()
                if len(parts) >= 5 and re.fullmatch(r"\d+", parts[0]):
                    moms.append(float(parts[-1]))
                j += 1
            if moms:
                last_block = moms
            i = j
        else:
            i += 1
    return np.array(last_block, dtype=np.float64) if last_block is not None else None


def resolve_spin_alignment(
    data: VaspVolumetricData,
    anchor_idx: Optional[int] = None,
    outcar_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Determine whether VASP block 0 (spin-up) or block 1 (spin-down) corresponds to
    local majority spin on the anchor / dominant magnetic atom.
    """
    if len(data.blocks) < 2:
        return {"is_spin_polarized": False, "majority_block": 0, "minority_block": 0, "source": "single_block"}

    if outcar_path is None:
        cand = os.path.join(os.path.dirname(data.filepath), "OUTCAR")
        if os.path.isfile(cand):
            outcar_path = cand

    if outcar_path and os.path.isfile(outcar_path):
        moms = read_outcar_magnetization(outcar_path)
        if moms is not None and len(moms) == data.natoms:
            idx = anchor_idx if anchor_idx is not None else int(np.argmax(np.abs(moms)))
            m_val = float(moms[idx])
            if abs(m_val) > 0.05:
                maj_idx = 0 if m_val > 0 else 1
                return {
                    "is_spin_polarized": True,
                    "majority_block": maj_idx,
                    "minority_block": 1 - maj_idx,
                    "anchor_label": data.atom_labels[idx],
                    "anchor_moment": round(m_val, 4),
                    "source": outcar_path,
                }

    # Fallback: check net difference around anchor or total integral
    if data.field_kind == "chg":
        # Block 1 is already (up - down)
        net_spin = float(np.sum(data.blocks[1]))
        maj_idx = 0 if net_spin >= 0 else 1
        return {"is_spin_polarized": True, "majority_block": maj_idx, "minority_block": 1 - maj_idx, "source": "net_spin_fallback"}

    # For ELFCAR/LOCPOT without OUTCAR: default block 0 = up, block 1 = down
    return {
        "is_spin_polarized": True,
        "majority_block": 0,
        "minority_block": 1,
        "source": "default_up_is_block0 (provide OUTCAR if local spin flips)",
    }


def mic_delta_frac(frac_a: np.ndarray, frac_b: np.ndarray) -> np.ndarray:
    """Minimum-image fractional vector from A to B in [-0.5, 0.5)."""
    df = (frac_b - frac_a) % 1.0
    df = np.where(df >= 0.5, df - 1.0, df)
    return df


def mic_distance(frac_a: np.ndarray, frac_b: np.ndarray, lattice: np.ndarray) -> float:
    df = mic_delta_frac(frac_a, frac_b)
    return float(np.linalg.norm(df @ lattice))


def unwrap_atoms_around_anchor(data: VaspVolumetricData, anchor_idx: int) -> np.ndarray:
    """
    Return Cartesian coordinates (N, 3) of all atoms shifted to their minimum-image
    representation relative to anchor_idx.
    """
    f0 = data.frac_coords[anchor_idx]
    df = mic_delta_frac(f0[None, :], data.frac_coords)
    return (f0[None, :] + df) @ data.lattice


def detect_bonds_mic(
    data: VaspVolumetricData,
    scale_tol: float = 1.25,
    max_bond_A: float = 3.2,
    allowed_pairs: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[int, int, float]]:
    """
    Detect chemical bonds across periodic boundaries using covalent radii + tolerance.
    Returns list of (idx_i, idx_j, distance_A) with idx_i < idx_j.
    """
    bonds: List[Tuple[int, int, float]] = []
    n = data.natoms
    lat = data.lattice
    frac = data.frac_coords
    elems = data.atom_elements

    pair_set = None
    if allowed_pairs:
        pair_set = {tuple(sorted(p)) for p in allowed_pairs}

    for i in range(n):
        ri = COVALENT_RADII.get(elems[i], 1.40)
        for j in range(i + 1, n):
            if pair_set is not None:
                key = tuple(sorted((elems[i], elems[j])))
                if key not in pair_set:
                    continue
            rj = COVALENT_RADII.get(elems[j], 1.40)
            cutoff = min(max_bond_A, (ri + rj) * scale_tol)
            d = mic_distance(frac[i], frac[j], lat)
            if 0.4 < d <= cutoff:
                bonds.append((i, j, d))
    return bonds


def eval_grid_cartesian(
    grid: np.ndarray,
    lattice: np.ndarray,
    pts_cart: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """
    Evaluate a periodic 3D VASP volumetric grid at arbitrary Cartesian points (M, 3)
    using periodic cubic spline interpolation (mode='wrap').
    """
    inv_lat = np.linalg.inv(lattice)
    f = pts_cart @ inv_lat
    ng = np.array(grid.shape, dtype=np.float64)
    coords = np.vstack([
        (f[:, 0] % 1.0) * ng[0],
        (f[:, 1] % 1.0) * ng[1],
        (f[:, 2] % 1.0) * ng[2],
    ])
    return map_coordinates(grid, coords, order=order, mode="grid-wrap")


def parse_atom_selector(selector: str, data: VaspVolumetricData) -> List[int]:
    """
    Parse comma-separated atom selectors into 0-based indices.
    Supports:
      - 1-based integer indices: "1,5,12"
      - Atom labels: "Fe1,S1,S2" or "Fe1,O1,Cl2,O2"
      - Element names (selects all atoms of that element): "Fe,Cl"
    """
    tokens = [t.strip() for t in selector.split(",") if t.strip()]
    label_to_idx = {lbl.lower(): i for i, lbl in enumerate(data.atom_labels)}
    indices: List[int] = []
    for tok in tokens:
        if re.fullmatch(r"\d+", tok):
            idx = int(tok) - 1
            if not (0 <= idx < data.natoms):
                raise ValueError(f"Atom index {tok} out of range (1..{data.natoms})")
            indices.append(idx)
        elif tok.lower() in label_to_idx:
            indices.append(label_to_idx[tok.lower()])
        else:
            matched = [i for i, el in enumerate(data.atom_elements) if el.lower() == tok.lower()]
            if not matched:
                raise ValueError(f"Cannot resolve atom selector '{tok}' against labels {data.atom_labels[:10]}...")
            indices.extend(matched)
    # Preserve order while deduplicating
    seen = set()
    out: List[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out
