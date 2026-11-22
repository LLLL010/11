#!/usr/bin/env python3
"""
General Crystallographic & Molecular Slice Plane Selection.

Supports:
  - atoms         : Exact 3-atom plane or SVD best-fit plane through >=4 MIC-unwrapped atoms
  - hkl           : Crystallographic Miller plane (h k l) passing through an anchor atom/point
  - vertical-scan : Surface/slab vertical plane rotating around anchor to minimize atom residuals
  - auto          : Automatically distinguishes bulk crystals (e.g., pyrite FeS2) vs slabs vs molecules
                    and selects the highest-coplanarity motif or low-index plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from vasp_grid_io import (
    COVALENT_RADII,
    VaspVolumetricData,
    detect_bonds_mic,
    mic_delta_frac,
    parse_atom_selector,
    unwrap_atoms_around_anchor,
)

TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
}


@dataclass
class SlicePlane:
    origin: np.ndarray       # (3,) Cartesian origin (u=0, v=0)
    e_u: np.ndarray          # (3,) unit vector along horizontal plot axis u
    e_v: np.ndarray          # (3,) unit vector along vertical plot axis v
    e_n: np.ndarray          # (3,) unit normal perpendicular to slice
    mode: str
    description: str
    anchor_idx: Optional[int]
    rms_residual: float
    target_residuals: Dict[str, float]


def classify_system_dimensionality(data: VaspVolumetricData, vacuum_thresh_A: float = 6.0) -> Tuple[str, Optional[int]]:
    """
    Classify structure as 'bulk', 'slab', or 'molecule' by checking fractional gaps along each axis.
    Returns (kind, vacuum_axis_idx_if_slab).
    """
    vacuum_axes: List[int] = []
    for ax in range(3):
        axis_len = float(np.linalg.norm(data.lattice[ax]))
        f_sorted = np.sort(data.frac_coords[:, ax] % 1.0)
        if len(f_sorted) <= 1:
            vacuum_axes.append(ax)
            continue
        gaps = np.diff(np.concatenate([f_sorted, [f_sorted[0] + 1.0]]))
        max_gap_A = float(np.max(gaps) * axis_len)
        if max_gap_A >= vacuum_thresh_A:
            vacuum_axes.append(ax)

    if len(vacuum_axes) == 0:
        return "bulk", None
    if len(vacuum_axes) == 1:
        return "slab", vacuum_axes[0]
    return "molecule", None


def _orthonormal_basis_from_normal(
    e_n: np.ndarray,
    preferred_v: Optional[np.ndarray] = None,
    preferred_u: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_n = e_n / np.linalg.norm(e_n)

    if preferred_u is not None:
        u_proj = preferred_u - np.dot(preferred_u, e_n) * e_n
        if np.linalg.norm(u_proj) > 1e-4:
            e_u = u_proj / np.linalg.norm(u_proj)
            e_v = np.cross(e_n, e_u)
            e_v /= np.linalg.norm(e_v)
            return e_u, e_v, e_n

    if preferred_v is None:
        preferred_v = np.array([0.0, 0.0, 1.0])
    v_proj = preferred_v - np.dot(preferred_v, e_n) * e_n
    if np.linalg.norm(v_proj) < 1e-4:
        # Normal is nearly parallel to z; use +y as vertical, +x as horizontal
        for cand in [np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]:
            v_proj = cand - np.dot(cand, e_n) * e_n
            if np.linalg.norm(v_proj) > 1e-4:
                break
    e_v = v_proj / np.linalg.norm(v_proj)
    # Right-handed or clean screen orientation: e_u = e_v x e_n
    e_u = np.cross(e_v, e_n)
    e_u /= np.linalg.norm(e_u)
    return e_u, e_v, e_n


def _pick_default_anchor(data: VaspVolumetricData, kind: str, vac_axis: Optional[int]) -> int:
    """Pick a chemically sensible default anchor atom (transition metal / rare heteroatom)."""
    counts_by_el: Dict[str, int] = dict(zip(data.elements, data.counts))
    # Priority 1: rarest transition metal
    tm_indices = [i for i, el in enumerate(data.atom_elements) if el in TRANSITION_METALS]
    if tm_indices:
        if kind == "slab" and vac_axis is not None:
            # Prefer surface-exposed transition metal (highest along slab normal if multiple)
            return max(tm_indices, key=lambda i: (-counts_by_el[data.atom_elements[i]], data.cart_coords[i, vac_axis]))
        return min(tm_indices, key=lambda i: (counts_by_el[data.atom_elements[i]], i))
    # Priority 2: rarest non-H/non-C heteroatom
    hetero = [i for i, el in enumerate(data.atom_elements) if el not in ("H", "C")]
    if hetero:
        return min(hetero, key=lambda i: (counts_by_el[data.atom_elements[i]], i))
    return 0


def build_plane_from_atoms(
    data: VaspVolumetricData,
    atom_indices: List[int],
    anchor_idx: Optional[int] = None,
    slab_normal: Optional[np.ndarray] = None,
) -> SlicePlane:
    if len(atom_indices) < 3:
        raise ValueError("Need at least 3 non-collinear atoms for 'atoms' mode.")
    a_idx = anchor_idx if anchor_idx is not None else atom_indices[0]
    unwrapped = unwrap_atoms_around_anchor(data, a_idx)
    pts = unwrapped[atom_indices]
    origin = unwrapped[a_idx].copy()

    if len(atom_indices) == 3:
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) < 1e-5:
            raise ValueError("Selected 3 atoms are collinear; cannot define a unique plane.")
        e_n = cross / np.linalg.norm(cross)
        pref_u = v1 if slab_normal is None else None
        e_u, e_v, e_n = _orthonormal_basis_from_normal(e_n, preferred_v=slab_normal, preferred_u=pref_u)
    else:
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - center)
        e_n = vh[-1]
        e_u, e_v, e_n = _orthonormal_basis_from_normal(e_n, preferred_v=slab_normal)
        # Keep origin on the plane near anchor
        origin = origin - np.dot(origin - center, e_n) * e_n

    res = {data.atom_labels[idx]: round(float(np.dot(unwrapped[idx] - origin, e_n)), 4) for idx in atom_indices}
    rms = float(np.sqrt(np.mean([v * v for v in res.values()])))
    labels_str = ",".join(data.atom_labels[i] for i in atom_indices)
    return SlicePlane(
        origin=origin,
        e_u=e_u,
        e_v=e_v,
        e_n=e_n,
        mode="atoms",
        description=f"Best-fit plane through [{labels_str}]",
        anchor_idx=a_idx,
        rms_residual=rms,
        target_residuals=res,
    )


def build_plane_from_hkl(
    data: VaspVolumetricData,
    hkl: Tuple[int, int, int],
    anchor_idx: Optional[int] = None,
    origin_cart: Optional[np.ndarray] = None,
) -> SlicePlane:
    h, k, l = hkl
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices (h, k, l) cannot all be zero.")
    inv_lat = np.linalg.inv(data.lattice)
    n_vec = inv_lat @ np.array([h, k, l], dtype=np.float64)
    e_n = n_vec / np.linalg.norm(n_vec)

    if origin_cart is not None:
        origin = origin_cart.copy()
        a_idx = anchor_idx
    else:
        a_idx = anchor_idx if anchor_idx is not None else 0
        origin = data.cart_coords[a_idx].copy()

    # Prefer [001] as vertical axis unless plane normal is parallel to [001]
    c_dir = data.lattice[2] / np.linalg.norm(data.lattice[2])
    b_dir = data.lattice[1] / np.linalg.norm(data.lattice[1])
    pref_v = c_dir if abs(np.dot(e_n, c_dir)) < 0.95 else b_dir
    e_u, e_v, e_n = _orthonormal_basis_from_normal(e_n, preferred_v=pref_v)

    return SlicePlane(
        origin=origin,
        e_u=e_u,
        e_v=e_v,
        e_n=e_n,
        mode="hkl",
        description=f"({h}{k}{l}) plane through {data.atom_labels[a_idx] if a_idx is not None else 'origin'}",
        anchor_idx=a_idx,
        rms_residual=0.0,
        target_residuals={},
    )


def build_plane_vertical_scan(
    data: VaspVolumetricData,
    target_indices: List[int],
    anchor_idx: Optional[int] = None,
    normal_axis: Optional[np.ndarray] = None,
    fixed_theta_deg: Optional[float] = None,
    z_origin_mode: str = "anchor",
) -> SlicePlane:
    """
    Fit a vertical plane containing the surface normal (default [0,0,1]) passing through anchor_idx,
    by scanning azimuth theta in [0, 180) deg to minimize RMS perpendicular distance of target_indices.
    """
    a_idx = anchor_idx if anchor_idx is not None else target_indices[0]
    unwrapped = unwrap_atoms_around_anchor(data, a_idx)
    origin = unwrapped[a_idx].copy()

    if normal_axis is None:
        e_v = np.array([0.0, 0.0, 1.0])
    else:
        e_v = normal_axis / np.linalg.norm(normal_axis)

    # Orthonormal basis in plane perpendicular to e_v
    ref = np.array([1.0, 0.0, 0.0]) if abs(e_v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = ref - np.dot(ref, e_v) * e_v
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(e_v, b1)
    b2 /= np.linalg.norm(b2)

    if z_origin_mode == "substrate_mean":
        # Optional: shift vertical zero to substrate plane if C exists
        c_zs = [unwrapped[i, 2] for i, el in enumerate(data.atom_elements) if el == "C"]
        if c_zs:
            origin[2] = float(np.mean(c_zs))

    pts = unwrapped[target_indices] - origin[None, :]
    if fixed_theta_deg is not None:
        best_th = float(fixed_theta_deg)
    else:
        thetas = np.linspace(0.0, 180.0, 360, endpoint=False)
        best_rms = np.inf
        best_th = 0.0
        for th_deg in thetas:
            th = np.deg2rad(th_deg)
            n_cand = -np.sin(th) * b1 + np.cos(th) * b2
            dists = pts @ n_cand
            rms = float(np.sqrt(np.mean(dists * dists)))
            if rms < best_rms:
                best_rms = rms
                best_th = float(th_deg)

    th = np.deg2rad(best_th)
    e_u = np.cos(th) * b1 + np.sin(th) * b2
    e_n = -np.sin(th) * b1 + np.cos(th) * b2

    res = {data.atom_labels[idx]: round(float(np.dot(unwrapped[idx] - origin, e_n)), 4) for idx in target_indices}
    rms = float(np.sqrt(np.mean([v * v for v in res.values()])))

    return SlicePlane(
        origin=origin,
        e_u=e_u,
        e_v=e_v,
        e_n=e_n,
        mode="vertical-scan",
        description=f"Vertical slice (theta={best_th:.1f} deg) through {data.atom_labels[a_idx]}",
        anchor_idx=a_idx,
        rms_residual=rms,
        target_residuals=res,
    )


def _score_plane_on_periodic_crystal(
    data: VaspVolumetricData,
    plane: SlicePlane,
    bonds: List[Tuple[int, int, float]],
    tol_perp: float = 0.25,
    window_radius: float = 5.5,
) -> Tuple[int, int, float]:
    """
    Score a candidate plane by counting how many real periodic bonds & atoms lie within tol_perp
    inside the viewing window. Returns (coplanar_bond_count, coplanar_atom_count, -rms).
    """
    imgs_cart, imgs_idx = collect_periodic_images_in_slab(
        data, plane, u_lim=(-window_radius, window_radius), v_lim=(-window_radius, window_radius), max_d_perp=tol_perp
    )
    if len(imgs_cart) == 0:
        return 0, 0, -999.0

    # Build quick bond distance lookup by atom pair
    bond_cutoffs: Dict[Tuple[int, int], float] = {}
    for i, j, d in bonds:
        bond_cutoffs[(i, j)] = d + 0.08
        bond_cutoffs[(j, i)] = d + 0.08

    coplanar_bonds = 0
    for a in range(len(imgs_cart)):
        ia = imgs_idx[a]
        for b in range(a + 1, len(imgs_cart)):
            ib = imgs_idx[b]
            cut = bond_cutoffs.get((ia, ib))
            if cut is not None:
                dist = float(np.linalg.norm(imgs_cart[a] - imgs_cart[b]))
                if dist <= cut:
                    coplanar_bonds += 1

    dists_perp = (imgs_cart - plane.origin[None, :]) @ plane.e_n
    rms = float(np.sqrt(np.mean(dists_perp * dists_perp)))
    return coplanar_bonds, len(imgs_cart), -rms


def auto_select_plane(
    data: VaspVolumetricData,
    anchor_selector: Optional[str] = None,
    atom_selector: Optional[str] = None,
) -> Tuple[SlicePlane, List[Tuple[str, SlicePlane, Tuple[int, int, float]]]]:
    """
    Automatically select the best slice plane for any VASP system (bulk crystal, slab, or molecule).
    Returns (chosen_plane, ranked_candidates).
    """
    kind, vac_axis = classify_system_dimensionality(data)
    if anchor_selector:
        anchor_idx = parse_atom_selector(anchor_selector, data)[0]
    else:
        anchor_idx = _pick_default_anchor(data, kind, vac_axis)

    bonds = detect_bonds_mic(data)

    # If user explicitly supplied target atoms, decide between vertical-scan (slab) or 3D plane (bulk/mol)
    if atom_selector:
        targets = parse_atom_selector(atom_selector, data)
        if kind == "slab" and vac_axis is not None:
            slab_n = np.cross(data.lattice[(vac_axis + 1) % 3], data.lattice[(vac_axis + 2) % 3])
            slab_n /= np.linalg.norm(slab_n)
            if slab_n[2] < 0:
                slab_n = -slab_n
            plane = build_plane_vertical_scan(data, targets, anchor_idx=anchor_idx, normal_axis=slab_n)
        else:
            plane = build_plane_from_atoms(data, targets, anchor_idx=anchor_idx)
        return plane, [(plane.description, plane, (0, len(targets), -plane.rms_residual))]

    if kind == "slab" and vac_axis is not None:
        # Find active site cluster: anchor + bonded neighbors + 2nd-shell adsorbate neighbors
        adj: Dict[int, List[int]] = {i: [] for i in range(data.natoms)}
        for i, j, _ in bonds:
            adj[i].append(j)
            adj[j].append(i)

        slab_n = np.cross(data.lattice[(vac_axis + 1) % 3], data.lattice[(vac_axis + 2) % 3])
        slab_n /= np.linalg.norm(slab_n)
        if slab_n[2] < 0:
            slab_n = -slab_n

        # Prioritize non-substrate / out-of-plane active-site atoms (adsorbate + axial ligand)
        unwrapped = unwrap_atoms_around_anchor(data, anchor_idx)
        anchor_h = float(np.dot(unwrapped[anchor_idx], slab_n))
        out_of_plane_active = [anchor_idx]
        in_plane_neigh = []
        for j in range(data.natoms):
            if j == anchor_idx:
                continue
            d = float(np.linalg.norm(unwrapped[j] - unwrapped[anchor_idx]))
            dz = abs(float(np.dot(unwrapped[j] - unwrapped[anchor_idx], slab_n)))
            el_j = data.atom_elements[j]
            if d <= 5.0 and (el_j != "C" or j in adj[anchor_idx]):
                if dz >= 0.45:
                    out_of_plane_active.append(j)
                elif j in adj[anchor_idx] and el_j != "C":
                    in_plane_neigh.append(j)

        if len(out_of_plane_active) >= 2:
            nearby_active = out_of_plane_active
        else:
            nearby_active = out_of_plane_active + in_plane_neigh[:2]
        if len(nearby_active) < 2:
            nearby_active.extend(adj[anchor_idx][:4])
        nearby_active = list(dict.fromkeys(nearby_active))

        plane = build_plane_vertical_scan(data, nearby_active, anchor_idx=anchor_idx, normal_axis=slab_n)
        plane.mode = "auto(slab)"
        return plane, [(plane.description, plane, (len(nearby_active), len(nearby_active), -plane.rms_residual))]

    # Bulk crystal or molecule: evaluate low-index hkl planes + local motif 3-atom planes!
    candidates: List[Tuple[str, SlicePlane]] = []
    for hkl in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        p = build_plane_from_hkl(data, hkl, anchor_idx=anchor_idx)
        candidates.append((f"hkl={hkl}", p))

    # Also build local 3-atom motif planes around anchor (e.g., Fe-S-S dumbbell in pyrite or M-L1-L2)
    adj_b: Dict[int, List[int]] = {i: [] for i in range(data.natoms)}
    for i, j, _ in bonds:
        adj_b[i].append(j)
        adj_b[j].append(i)

    unwrapped = unwrap_atoms_around_anchor(data, anchor_idx)
    neigh1 = sorted(adj_b[anchor_idx], key=lambda j: float(np.linalg.norm(unwrapped[j] - unwrapped[anchor_idx])))
    # Motif 1: anchor + neighbor1 + second-shell neighbor (captures Fe-S-S dumbbell in pyrite!)
    motif_triples: List[Tuple[int, int, int]] = []
    for n1 in neigh1[:3]:
        for n2 in adj_b[n1]:
            if n2 != anchor_idx and n2 not in neigh1:
                motif_triples.append((anchor_idx, n1, n2))
                break
        if motif_triples:
            break
    # Motif 2: anchor + two non-collinear nearest neighbors (octahedral/tetrahedral plane)
    if len(neigh1) >= 2:
        for a_i in range(min(3, len(neigh1))):
            for b_i in range(a_i + 1, min(4, len(neigh1))):
                v1 = unwrapped[neigh1[a_i]] - unwrapped[anchor_idx]
                v2 = unwrapped[neigh1[b_i]] - unwrapped[anchor_idx]
                if np.linalg.norm(np.cross(v1, v2)) > 0.3:
                    motif_triples.append((anchor_idx, neigh1[a_i], neigh1[b_i]))
                    break
            if len(motif_triples) >= 2:
                break

    for t in motif_triples[:3]:
        try:
            p = build_plane_from_atoms(data, list(t), anchor_idx=anchor_idx)
            candidates.append((p.description, p))
        except ValueError:
            pass

    scored: List[Tuple[str, SlicePlane, Tuple[int, int, float]]] = []
    for name, p in candidates:
        sc = _score_plane_on_periodic_crystal(data, p, bonds)
        scored.append((name, p, sc))

    # Sort by coplanar bonds (desc), coplanar atoms (desc), negative rms (desc), and prefer hkl if tied
    scored.sort(key=lambda item: (item[2][0], item[2][1], round(item[2][2], 3), 1 if item[1].mode == "hkl" else 0), reverse=True)
    best_name, best_plane, _ = scored[0]
    best_plane.mode = f"auto({best_plane.mode})"
    return best_plane, scored


def collect_periodic_images_in_slab(
    data: VaspVolumetricData,
    plane: SlicePlane,
    u_lim: Tuple[float, float],
    v_lim: Tuple[float, float],
    max_d_perp: float = 0.45,
) -> Tuple[np.ndarray, List[int]]:
    """
    Find all periodic atomic images inside the 2D slice box [u_lim] x [v_lim] within |d_perp| <= max_d_perp.
    Returns (cart_positions (M, 3), original_atom_indices (M,)).
    """
    max_r = float(
        np.sqrt(max(abs(u_lim[0]), abs(u_lim[1])) ** 2 + max(abs(v_lim[0]), abs(v_lim[1])) ** 2 + max_d_perp ** 2)
    ) + 2.0

    # Determine required supercell range along each lattice vector
    inv_lat = np.linalg.inv(data.lattice)
    origin_frac = plane.origin @ inv_lat
    reps = []
    for ax in range(3):
        ax_len = float(np.linalg.norm(data.lattice[ax]))
        n_rep = max(1, int(np.ceil(max_r / max(ax_len, 1.0))) + 1)
        reps.append(range(-n_rep, n_rep + 1))

    # Build shift vectors
    shifts_frac = np.array(
        [[ix, iy, iz] for ix in reps[0] for iy in reps[1] for iz in reps[2]],
        dtype=np.float64,
    )
    shifts_cart = shifts_frac @ data.lattice  # (S, 3)

    # Shift base atoms close to origin_frac first
    base_frac = data.frac_coords - np.round(data.frac_coords - origin_frac[None, :])
    base_cart = base_frac @ data.lattice

    # Broadcast: (N, 1, 3) + (1, S, 3) -> (N*S, 3)
    all_cart = (base_cart[:, None, :] + shifts_cart[None, :, :]).reshape(-1, 3)
    all_idx = np.repeat(np.arange(data.natoms), len(shifts_cart))

    rel = all_cart - plane.origin[None, :]
    d_perp = rel @ plane.e_n
    mask = np.abs(d_perp) <= max_d_perp
    if not np.any(mask):
        return np.empty((0, 3)), []

    sub_cart = all_cart[mask]
    sub_rel = rel[mask]
    sub_idx = all_idx[mask]

    u = sub_rel @ plane.e_u
    v = sub_rel @ plane.e_v
    box_mask = (u >= u_lim[0] - 0.3) & (u <= u_lim[1] + 0.3) & (v >= v_lim[0] - 0.3) & (v <= v_lim[1] + 0.3)
    return sub_cart[box_mask], sub_idx[box_mask].tolist()
