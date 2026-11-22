#!/usr/bin/env python3
"""
VASP Volumetric Slicer, Bond Quantifier, and Publication-Grade Plotter.

Supports direct VASP output files (ELFCAR, CHGCAR, CHG, LOCPOT, PARCHG, OUTCAR)
for periodic bulk crystals (e.g., pyrite FeS2), slabs/interfaces, single-atom catalysts,
and molecular clusters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# Ensure local script imports work regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from plane_selector import (
    SlicePlane,
    auto_select_plane,
    build_plane_from_atoms,
    build_plane_from_hkl,
    build_plane_vertical_scan,
    classify_system_dimensionality,
    collect_periodic_images_in_slab,
)
from vasp_grid_io import (
    VaspVolumetricData,
    detect_bonds_mic,
    eval_grid_cartesian,
    mic_delta_frac,
    parse_atom_selector,
    read_vasp_volumetric,
    resolve_spin_alignment,
)

# Standard publication colormaps
ELF_CMAP = LinearSegmentedColormap.from_list(
    "vesta_elf",
    ["#00007F", "#0000FF", "#0080FF", "#00FFFF", "#00FF80", "#80FF00", "#FFFF00", "#FF8000", "#D90000"],
    N=256,
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "pub_diverging",
    ["#1E3A8A", "#3B82F6", "#BFDBFE", "#F8FAFC", "#FECACA", "#EF4444", "#991B1B"],
    N=256,
)


def configure_matplotlib() -> None:
    """Set journal-standard typography and editable vector export settings."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.85,
        "axes.edgecolor": "#1A1A1A",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,      # TrueType editable text in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # Pure <text> elements in SVG
        "savefig.transparent": False,
    })


def resolve_input_file(path: str, field_hint: str) -> str:
    """If path is a VASP run directory, locate the appropriate volumetric file."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input path does not exist: {path}")

    priority = {
        "elf": ["ELFCAR", "ELFCAR.vasp"],
        "chg": ["CHGCAR", "CHG"],
        "spin": ["CHGCAR"],
        "locpot": ["LOCPOT"],
        "parchg": ["PARCHG"],
        "cdd": ["CHGDIFF.vasp", "CHGDIFF", "CHGCAR"],
        "auto": ["ELFCAR", "CHGCAR", "LOCPOT", "PARCHG", "CHG"],
    }
    for fname in priority.get(field_hint, priority["auto"]):
        cand = os.path.join(path, fname)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"No VASP volumetric file ({priority.get(field_hint)}) found in {path}")


def extract_field_grids(
    data: VaspVolumetricData,
    field: str,
    spin_info: Dict[str, object],
) -> Tuple[str, Dict[str, np.ndarray], str]:
    """
    Convert raw VASP blocks into physical grids with proper units.
    Returns (resolved_field, grids_dict, colorbar_unit_label).
    """
    if field == "auto":
        field = data.field_kind if data.field_kind != "unknown" else "chg"

    vol = data.volume
    grids: Dict[str, np.ndarray] = {}

    if field == "elf":
        if len(data.blocks) == 1:
            grids["avg"] = np.clip(data.blocks[0], 0.0, 1.0)
        else:
            maj_i = int(spin_info["majority_block"])
            min_i = int(spin_info["minority_block"])
            b_up = np.clip(data.blocks[0], 0.0, 1.0)
            b_dn = np.clip(data.blocks[1], 0.0, 1.0)
            b_maj = np.clip(data.blocks[maj_i], 0.0, 1.0)
            b_min = np.clip(data.blocks[min_i], 0.0, 1.0)
            grids["up"] = b_up
            grids["down"] = b_dn
            grids["maj"] = b_maj
            grids["min"] = b_min
            grids["avg"] = 0.5 * (b_maj + b_min)
            grids["diff"] = b_maj - b_min
        return "elf", grids, "ELF"

    if field == "chg":
        grids["avg"] = data.blocks[0] / vol
        if len(data.blocks) >= 2:
            # Block 1 in VASP CHGCAR is spin density (up - down) * V
            spin_raw = data.blocks[1] / vol
            sign = 1.0 if int(spin_info["majority_block"]) == 0 else -1.0
            grids["spin"] = sign * spin_raw
        return "chg", grids, r"$\rho$ ($e\ \mathrm{\AA}^{-3}$)"

    if field == "spin":
        if len(data.blocks) < 2:
            raise ValueError(f"{data.filepath} only has 1 volumetric block; cannot extract spin density.")
        spin_raw = data.blocks[1] / vol
        sign = 1.0 if int(spin_info["majority_block"]) == 0 else -1.0
        grids["diff"] = sign * spin_raw
        return "spin", grids, r"$\rho_{\mathrm{spin}}$ ($\mu_{\mathrm{B}}\ \mathrm{\AA}^{-3}$)"

    if field == "cdd":
        grids["diff"] = data.blocks[0] / vol
        return "cdd", grids, r"$\Delta\rho$ ($e\ \mathrm{\AA}^{-3}$)"

    if field == "locpot":
        if len(data.blocks) == 1:
            grids["avg"] = data.blocks[0]
        else:
            maj_i = int(spin_info["majority_block"])
            min_i = int(spin_info["minority_block"])
            grids["maj"] = data.blocks[maj_i]
            grids["min"] = data.blocks[min_i]
            grids["avg"] = 0.5 * (data.blocks[0] + data.blocks[1])
            grids["diff"] = data.blocks[maj_i] - data.blocks[min_i]
        return "locpot", grids, r"$V_{\mathrm{loc}}$ (eV)"

    # parchg / unknown
    grids["avg"] = data.blocks[0] / vol
    return field, grids, "Density (a.u.)"


def build_plane_for_system(
    data: VaspVolumetricData,
    args: argparse.Namespace,
) -> Tuple[SlicePlane, List]:
    kind, vac_axis = classify_system_dimensionality(data)
    anchor_idx = parse_atom_selector(args.anchor, data)[0] if args.anchor else None

    if args.mode == "auto":
        return auto_select_plane(data, anchor_selector=args.anchor, atom_selector=args.atoms)

    if args.mode == "hkl":
        if args.hkl is None:
            raise ValueError("--hkl H K L is required when --mode hkl")
        p = build_plane_from_hkl(data, tuple(args.hkl), anchor_idx=anchor_idx)
        return p, [(p.description, p, (0, 0, 0.0))]

    if args.mode == "atoms":
        if not args.atoms:
            raise ValueError("--atoms is required when --mode atoms")
        targets = parse_atom_selector(args.atoms, data)
        slab_n = None
        if kind == "slab" and vac_axis is not None and not args.free_3d:
            slab_n = np.cross(data.lattice[(vac_axis + 1) % 3], data.lattice[(vac_axis + 2) % 3])
            slab_n /= np.linalg.norm(slab_n)
            if slab_n[2] < 0:
                slab_n = -slab_n
        p = build_plane_from_atoms(data, targets, anchor_idx=anchor_idx, slab_normal=slab_n)
        return p, [(p.description, p, (0, len(targets), -p.rms_residual))]

    if args.mode == "vertical-scan":
        if not args.atoms:
            raise ValueError("--atoms is required when --mode vertical-scan")
        targets = parse_atom_selector(args.atoms, data)
        p = build_plane_vertical_scan(
            data, targets, anchor_idx=anchor_idx, fixed_theta_deg=args.theta
        )
        return p, [(p.description, p, (0, len(targets), -p.rms_residual))]

    raise ValueError(f"Unsupported mode: {args.mode}")


def quantify_bonds_near_slice(
    data: VaspVolumetricData,
    grids: Dict[str, np.ndarray],
    plane: SlicePlane,
    u_lim: Tuple[float, float],
    v_lim: Tuple[float, float],
    max_d_perp: float,
    bond_filter: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Collect periodic atom projections and bond segments inside the slice window,
    plus exact 3D MIC quantitative line-profile statistics along every unique bond type/pair.
    """
    imgs_cart, imgs_idx = collect_periodic_images_in_slab(
        data, plane, u_lim=u_lim, v_lim=v_lim, max_d_perp=max_d_perp
    )
    rel = imgs_cart - plane.origin[None, :] if len(imgs_cart) > 0 else np.empty((0, 3))
    atom_records = []
    for k, orig_i in enumerate(imgs_idx):
        u = float(rel[k] @ plane.e_u)
        v = float(rel[k] @ plane.e_v)
        dn = float(rel[k] @ plane.e_n)
        atom_records.append({
            "orig_index": orig_i + 1,
            "label": data.atom_labels[orig_i],
            "element": data.atom_elements[orig_i],
            "u": round(u, 4),
            "v": round(v, 4),
            "d_perp": round(dn, 4),
            "x": round(float(imgs_cart[k, 0]), 4),
            "y": round(float(imgs_cart[k, 1]), 4),
            "z": round(float(imgs_cart[k, 2]), 4),
        })
    df_atoms = pd.DataFrame(atom_records)

    # Detect bonds among projected periodic images so periodic crystal bonds are never broken
    all_mic_bonds = detect_bonds_mic(data, allowed_pairs=bond_filter)
    mic_pair_len = {}
    for i, j, d in all_mic_bonds:
        mic_pair_len[(i, j)] = d
        mic_pair_len[(j, i)] = d

    seg_records = []
    bond_stats_records = []
    seen_unique_bonds = set()

    for a in range(len(imgs_idx)):
        ia = imgs_idx[a]
        pa = imgs_cart[a]
        ua, va = atom_records[a]["u"], atom_records[a]["v"]
        for b in range(a + 1, len(imgs_idx)):
            ib = imgs_idx[b]
            pb = imgs_cart[b]
            d_ref = mic_pair_len.get((ia, ib))
            if d_ref is None:
                continue
            dist = float(np.linalg.norm(pb - pa))
            if dist <= d_ref + 0.08:
                ub, vb = atom_records[b]["u"], atom_records[b]["v"]
                seg_records.append({
                    "atom_a": data.atom_labels[ia],
                    "atom_b": data.atom_labels[ib],
                    "u1": ua, "v1": va, "u2": ub, "v2": vb,
                    "length_A": round(dist, 4),
                })

                # Quantify exact 3D line profile once per unique (ia, ib) pair (nearest to origin)
                pair_key = (min(ia, ib), max(ia, ib))
                mid_dist_origin = float(np.linalg.norm(0.5 * (pa + pb) - plane.origin))
                if pair_key not in seen_unique_bonds:
                    seen_unique_bonds.add(pair_key)
                    t = np.linspace(0.0, 1.0, 201)
                    pts_line = pa[None, :] + t[:, None] * (pb - pa)[None, :]
                    mask_core = (t >= 0.20) & (t <= 0.80)
                    mask_basin = (t >= 0.25) & (t <= 0.75)

                    row = {
                        "bond": f"{data.atom_labels[ia]}-{data.atom_labels[ib]}",
                        "elements": f"{data.atom_elements[ia]}-{data.atom_elements[ib]}",
                        "length_A": round(dist, 4),
                        "mid_d_perp_A": round(float(0.5 * (atom_records[a]["d_perp"] + atom_records[b]["d_perp"])), 4),
                        "_dist_origin": mid_dist_origin,
                    }
                    for gname, gdata in grids.items():
                        vals = eval_grid_cartesian(gdata, data.lattice, pts_line)
                        row[f"{gname}_mid"] = round(float(vals[100]), 4)
                        row[f"{gname}_saddle_min"] = round(float(np.min(vals[mask_core])), 4)
                        row[f"{gname}_basin_max"] = round(float(np.max(vals[mask_basin])), 4)
                    bond_stats_records.append(row)

    df_segs = pd.DataFrame(seg_records)
    df_stats = pd.DataFrame(bond_stats_records)
    if not df_stats.empty:
        df_stats = df_stats.sort_values("_dist_origin").drop(columns=["_dist_origin"]).reset_index(drop=True)
    return df_atoms, df_segs, df_stats


def sample_2d_slice(
    data: VaspVolumetricData,
    grid: np.ndarray,
    plane: SlicePlane,
    u_lim: Tuple[float, float],
    v_lim: Tuple[float, float],
    nu: int = 320,
    nv: int = 320,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    us = np.linspace(u_lim[0], u_lim[1], nu)
    vs = np.linspace(v_lim[0], v_lim[1], nv)
    UU, VV = np.meshgrid(us, vs)
    pts = (
        plane.origin[None, None, :]
        + UU[:, :, None] * plane.e_u[None, None, :]
        + VV[:, :, None] * plane.e_v[None, None, :]
    )
    val = eval_grid_cartesian(grid, data.lattice, pts.reshape(-1, 3)).reshape(nv, nu)
    return us, vs, val


def _select_labels_for_plotting(df_atoms: pd.DataFrame, label_mode: str) -> pd.DataFrame:
    if df_atoms.empty or label_mode == "none":
        return df_atoms.iloc[0:0]
    if label_mode == "all":
        return df_atoms
    # "clean": keep atoms close to center / representative motif, avoid crowded edges
    df = df_atoms.copy()
    df["r_center"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)
    df = df.sort_values(["r_center", "d_perp"])
    kept_rows = []
    kept_pts: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        u, v = float(row["u"]), float(row["v"])
        # Skip labels too close to already labeled atom or right on frame border
        if any((u - ku) ** 2 + (v - kv) ** 2 < 0.75 ** 2 for ku, kv in kept_pts):
            continue
        if len(kept_rows) >= 12:
            break
        kept_rows.append(row)
        kept_pts.append((u, v))
    return pd.DataFrame(kept_rows)


def render_panel(
    ax: plt.Axes,
    us: np.ndarray,
    vs: np.ndarray,
    val: np.ndarray,
    df_atoms: pd.DataFrame,
    df_segs: pd.DataFrame,
    field: str,
    channel: str,
    panel_title: str,
    u_lim: Tuple[float, float],
    v_lim: Tuple[float, float],
    vmin: float,
    vmax: float,
    show_x: bool = True,
    show_y: bool = True,
    label_mode: str = "clean",
):
    is_div = (channel == "diff") or (field in ("spin", "cdd"))
    cmap = DIV_CMAP if is_div else (ELF_CMAP if field == "elf" else ("cividis" if field == "locpot" else "viridis"))
    val_plot = np.clip(val, vmin, vmax)

    im = ax.imshow(
        val_plot,
        extent=[u_lim[0], u_lim[1], v_lim[0], v_lim[1]],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )

    dark_bg = (field == "elf" and not is_div) or (field in ("chg", "locpot") and not is_div)
    bond_col = "white" if dark_bg else "#334155"
    bond_alpha = 0.78 if dark_bg else 0.60
    txt_col = "white" if dark_bg else "#111111"

    # Subtle bond skeleton
    if not df_segs.empty:
        for _, s in df_segs.iterrows():
            ax.plot(
                [s["u1"], s["u2"]],
                [s["v1"], s["v2"]],
                linestyle="--",
                linewidth=0.80,
                color=bond_col,
                alpha=bond_alpha,
                zorder=2,
            )

    # Restrained atom markers (inside plot bounds)
    if not df_atoms.empty:
        in_box = df_atoms[
            (df_atoms["u"] >= u_lim[0] + 0.1) & (df_atoms["u"] <= u_lim[1] - 0.1) &
            (df_atoms["v"] >= v_lim[0] + 0.1) & (df_atoms["v"] <= v_lim[1] - 0.1)
        ]
        if not in_box.empty:
            ax.scatter(
                in_box["u"], in_box["v"],
                s=22, facecolor="white", edgecolor="#111111", linewidth=0.85, zorder=3
            )
            lbl_df = _select_labels_for_plotting(in_box, label_mode)
            du_span = u_lim[1] - u_lim[0]
            dv_span = v_lim[1] - v_lim[0]
            for _, r in lbl_df.iterrows():
                u, v = float(r["u"]), float(r["v"])
                # Offset slightly away from atom marker
                dx = 0.025 * du_span if u < u_lim[1] - 0.12 * du_span else -0.025 * du_span
                dy = 0.025 * dv_span if v < v_lim[1] - 0.12 * dv_span else -0.025 * dv_span
                ha = "left" if dx > 0 else "right"
                va = "bottom" if dy > 0 else "top"
                stroke_col = "#0F172A" if dark_bg else "#FFFFFF"
                ax.text(
                    u + dx, v + dy, str(r["element"]),
                    color=txt_col, fontsize=8.0, fontweight="bold",
                    ha=ha, va=va, zorder=4,
                    path_effects=[patheffects.withStroke(linewidth=1.8, foreground=stroke_col)],
                )

    # Clean panel title above frame (standard journal convention, zero internal clutter)
    ax.set_title(panel_title, loc="left", fontsize=9.5, fontweight="bold", pad=5.0, color="#111111")
    ax.set_xlim(u_lim)
    ax.set_ylim(v_lim)

    if show_x:
        ax.set_xlabel("In-plane distance (Å)", fontsize=9.0, fontweight="bold")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_y:
        ax.set_ylabel("Vertical distance (Å)", fontsize=9.0, fontweight="bold")
    else:
        ax.tick_params(axis="y", labelleft=False)

    return im


def run_pipeline(args: argparse.Namespace) -> None:
    configure_matplotlib()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load datasets (either standard files or CDD subtraction AB - A - B)
    systems: List[Tuple[str, VaspVolumetricData]] = []
    if args.cdd_ab:
        if not (args.cdd_a and args.cdd_b):
            raise ValueError("--cdd-ab requires both --cdd-a and --cdd-b")
        d_ab = read_vasp_volumetric(resolve_input_file(args.cdd_ab, "chg"), max_blocks=1)
        d_a = read_vasp_volumetric(resolve_input_file(args.cdd_a, "chg"), max_blocks=1)
        d_b = read_vasp_volumetric(resolve_input_file(args.cdd_b, "chg"), max_blocks=1)
        if d_ab.grid_shape != d_a.grid_shape or d_ab.grid_shape != d_b.grid_shape:
            raise ValueError(f"Grid mismatch in CDD subtraction: {d_ab.grid_shape} vs {d_a.grid_shape} vs {d_b.grid_shape}")
        d_ab.blocks = [d_ab.blocks[0] - d_a.blocks[0] - d_b.blocks[0]]
        d_ab.field_kind = "cdd"
        sys_name = args.names[0] if args.names else "CDD_System"
        systems.append((sys_name, d_ab))
    else:
        for idx, inp in enumerate(args.input):
            fpath = resolve_input_file(inp, args.field)
            data = read_vasp_volumetric(fpath, max_blocks=2)
            default_name = os.path.basename(os.path.dirname(os.path.abspath(fpath))) if os.path.basename(fpath).upper() in ("ELFCAR", "CHGCAR", "LOCPOT") else os.path.splitext(os.path.basename(fpath))[0]
            name = args.names[idx] if (args.names and idx < len(args.names)) else default_name
            systems.append((name, data))

    bond_filter = None
    if args.bonds and args.bonds.lower() != "auto":
        bond_filter = []
        for tok in args.bonds.split(","):
            parts = tok.strip().split("-")
            if len(parts) == 2:
                bond_filter.append((parts[0].strip(), parts[1].strip()))

    u_lim = (-args.width / 2.0, args.width / 2.0)
    v_lim = (-args.height / 2.0, args.height / 2.0)

    prepared = []
    all_atoms_dfs = []
    all_bonds_dfs = []
    all_slice_dfs = []
    diagnostics = {}

    for sys_name, data in systems:
        kind, vac_ax = classify_system_dimensionality(data)
        plane, ranked = build_plane_for_system(data, args)
        spin_info = resolve_spin_alignment(data, anchor_idx=plane.anchor_idx)
        field_resolved, grids, cbar_label = extract_field_grids(data, args.field, spin_info)

        df_atoms, df_segs, df_bonds = quantify_bonds_near_slice(
            data, grids, plane, u_lim=u_lim, v_lim=v_lim, max_d_perp=args.slab_tol, bond_filter=bond_filter
        )
        if not df_atoms.empty:
            df_atoms.insert(0, "system", sys_name)
            all_atoms_dfs.append(df_atoms)
        if not df_bonds.empty:
            df_bonds.insert(0, "system", sys_name)
            all_bonds_dfs.append(df_bonds)

        diagnostics[sys_name] = {
            "filepath": data.filepath,
            "dimensionality": kind,
            "field": field_resolved,
            "grid_shape": list(data.grid_shape),
            "spin_alignment": spin_info,
            "chosen_plane": {
                "mode": plane.mode,
                "description": plane.description,
                "origin_cart": np.round(plane.origin, 4).tolist(),
                "e_u": np.round(plane.e_u, 4).tolist(),
                "e_v": np.round(plane.e_v, 4).tolist(),
                "e_n": np.round(plane.e_n, 4).tolist(),
                "rms_residual_A": round(plane.rms_residual, 4),
                "target_residuals_A": plane.target_residuals,
            },
            "candidate_planes": [
                {"name": nm, "coplanar_bonds": int(sc[0]), "coplanar_atoms": int(sc[1]), "rms_A": round(float(-sc[2]), 4)}
                for nm, _, sc in ranked[:5]
            ],
        }

        print(f"\n=== [{sys_name}] ({kind}, field={field_resolved}, grid={data.grid_shape}) ===")
        print(f"  Plane: {plane.description} | RMS={plane.rms_residual:.4f} A")
        print(f"  Spin:  {spin_info}")
        if len(ranked) > 1:
            print("  Top candidate planes:")
            for nm, _, sc in ranked[:4]:
                print(f"    - {nm}: coplanar_bonds={sc[0]}, coplanar_atoms={sc[1]}, rms={-sc[2]:.4f} A")
        if not df_bonds.empty:
            print("  Bond midpoints / saddle stats (3D MIC):")
            print(df_bonds.head(12).to_string(index=False))

        prepared.append((sys_name, data, plane, field_resolved, grids, cbar_label, df_atoms, df_segs))

    with open(os.path.join(args.out_dir, f"{args.prefix}_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    if all_atoms_dfs:
        pd.concat(all_atoms_dfs, ignore_index=True).to_csv(
            os.path.join(args.out_dir, f"{args.prefix}_atoms.csv"), index=False
        )
    if all_bonds_dfs:
        pd.concat(all_bonds_dfs, ignore_index=True).to_csv(
            os.path.join(args.out_dir, f"{args.prefix}_bonds.csv"), index=False
        )

    if args.inspect_only:
        print("\n[Inspect-only mode] Skipped figure rendering.")
        return

    # Determine channels to render
    n_sys = len(prepared)
    first_field = prepared[0][3]
    first_grids = prepared[0][4]

    if args.spin == "panel":
        required = ["maj", "min", "diff"]
        if not all(k in p[4] for p in prepared for k in required):
            raise ValueError("--spin panel requires 2-block spin-polarized data (e.g. ISPIN=2 ELFCAR) for all systems.")
        # Render multi-row spin panel: rows=systems, cols=(Maj, Min, Diff)
        fig, axes = plt.subplots(
            n_sys, 3,
            figsize=(9.6, 3.05 * n_sys),
            squeeze=False,
            constrained_layout=True,
        )
        # Compute symmetric diff limit across all systems
        diff_vals = []
        for _, data, plane, _, grids, _, _, _ in prepared:
            _, _, dv = sample_2d_slice(data, grids["diff"], plane, u_lim, v_lim, args.grid_res, args.grid_res)
            diff_vals.append(dv)
        dmax = args.vlim if args.vlim else float(np.percentile(np.abs(np.concatenate([x.ravel() for x in diff_vals])), 99.5))
        dmax = max(dmax, 0.05)

        tags = [f"({chr(97 + i)})" for i in range(n_sys * 3)]
        t_idx = 0
        im_elf = None
        im_diff = None
        for r_i, (sys_name, data, plane, fld, grids, _, df_atoms, df_segs) in enumerate(prepared):
            for c_i, (ch, ch_lbl) in enumerate([("maj", "Maj."), ("min", "Min."), ("diff", "Maj. − Min.")]):
                us, vs, val = sample_2d_slice(data, grids[ch], plane, u_lim, v_lim, args.grid_res, args.grid_res)
                if ch == "diff":
                    vmin, vmax = -dmax, dmax
                elif fld == "elf":
                    vmin, vmax = 0.0, 1.0
                else:
                    vmin, vmax = float(np.percentile(val, 1.0)), float(np.percentile(val, 99.0))
                im = render_panel(
                    axes[r_i, c_i], us, vs, val, df_atoms, df_segs, fld, ch,
                    f"{tags[t_idx]} {sys_name} ({ch_lbl})",
                    u_lim, v_lim, vmin, vmax,
                    show_x=(r_i == n_sys - 1), show_y=(c_i == 0), label_mode=args.label_mode,
                )
                if ch == "diff":
                    im_diff = im
                else:
                    im_elf = im
                t_idx += 1

        fig.colorbar(im_elf, ax=axes[:, :2], fraction=0.024, pad=0.02, label=prepared[0][5])
        fig.colorbar(im_diff, ax=axes[:, 2], fraction=0.048, pad=0.03, label=r"$\Delta\mathrm{ELF}_{\mathrm{spin}}$")

        for ext in ("png", "pdf", "svg"):
            out_p = os.path.join(args.out_dir, f"{args.prefix}_spin_panel.{ext}")
            fig.savefig(out_p, dpi=600 if ext == "png" else None, facecolor="white")
        plt.close(fig)
        print(f"Saved spin panel figures to {args.out_dir}/{args.prefix}_spin_panel.(png/pdf/svg)")
        return

    # Standard side-by-side main figure
    ch = args.spin if args.spin in first_grids else ("avg" if "avg" in first_grids else "diff")
    sampled = []
    for sys_name, data, plane, fld, grids, cbar_lbl, df_atoms, df_segs in prepared:
        use_ch = ch if ch in grids else list(grids.keys())[0]
        us, vs, val = sample_2d_slice(data, grids[use_ch], plane, u_lim, v_lim, args.grid_res, args.grid_res)
        sampled.append((sys_name, fld, use_ch, cbar_lbl, us, vs, val, df_atoms, df_segs))
        if args.save_csv:
            UU, VV = np.meshgrid(us, vs)
            all_slice_dfs.append(pd.DataFrame({
                "system": sys_name, "u": UU.ravel(), "v": VV.ravel(), f"{fld}_{use_ch}": np.round(val.ravel(), 5)
            }))

    if args.save_csv and all_slice_dfs:
        pd.concat(all_slice_dfs, ignore_index=True).to_csv(
            os.path.join(args.out_dir, f"{args.prefix}_slice.csv"), index=False
        )

    # Shared color limits across panels for fair scientific comparison
    all_v = np.concatenate([s[6].ravel() for s in sampled])
    is_div = (sampled[0][2] == "diff") or (first_field in ("spin", "cdd"))
    if args.vlim is not None:
        vmin, vmax = (-args.vlim, args.vlim) if is_div else (0.0, args.vlim)
    elif first_field == "elf" and not is_div:
        vmin, vmax = 0.0, 1.0
    elif is_div:
        lim = max(float(np.percentile(np.abs(all_v), 99.2)), 1e-4)
        vmin, vmax = -lim, lim
    else:
        vmin, vmax = float(np.percentile(all_v, 1.0)), float(np.percentile(all_v, 99.0))

    fig_w = 3.2 * n_sys + 0.75
    fig_h = 3.25
    fig, axes = plt.subplots(1, n_sys, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)

    im_last = None
    for idx, (sys_name, fld, use_ch, cbar_lbl, us, vs, val, df_atoms, df_segs) in enumerate(sampled):
        tag = f"({chr(97 + idx)}) " if n_sys > 1 else ""
        im_last = render_panel(
            axes[0, idx], us, vs, val, df_atoms, df_segs, fld, use_ch,
            f"{tag}{sys_name}",
            u_lim, v_lim, vmin, vmax,
            show_x=True, show_y=(idx == 0), label_mode=args.label_mode,
        )

    cbar = fig.colorbar(im_last, ax=axes[0, :], fraction=0.032 / max(n_sys, 1) * 1.4, pad=0.025)
    cbar.set_label(sampled[0][3], fontsize=9.0, fontweight="bold")
    cbar.ax.tick_params(labelsize=8.0)

    for ext in ("png", "pdf", "svg"):
        out_p = os.path.join(args.out_dir, f"{args.prefix}.{ext}")
        fig.savefig(out_p, dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"Saved figures to {args.out_dir}/{args.prefix}.(png/pdf/svg)")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="General VASP Volumetric Slicer & Bond Quantifier")
    p.add_argument("--input", nargs="+", default=[], help="VASP output file(s) or run folder(s) (ELFCAR, CHGCAR, LOCPOT...)")
    p.add_argument("--names", nargs="+", default=None, help="Display names for each input system")
    p.add_argument("--field", default="auto", choices=["auto", "elf", "chg", "spin", "cdd", "locpot", "parchg"])
    p.add_argument("--spin", default="avg", choices=["avg", "up", "down", "majority", "minority", "diff", "panel"])
    p.add_argument("--cdd-ab", default=None, help="CHGCAR of AB complex for direct CDD subtraction")
    p.add_argument("--cdd-a", default=None, help="CHGCAR of substrate A")
    p.add_argument("--cdd-b", default=None, help="CHGCAR of adsorbate B")

    p.add_argument("--mode", default="auto", choices=["auto", "atoms", "hkl", "vertical-scan"])
    p.add_argument("--atoms", default=None, help="Comma-separated atom labels or 1-based indices (e.g. Fe1,S1,S2)")
    p.add_argument("--anchor", default=None, help="Anchor atom label or 1-based index (e.g. Fe1)")
    p.add_argument("--hkl", nargs=3, type=int, default=None, help="Miller indices H K L for --mode hkl")
    p.add_argument("--theta", type=float, default=None, help="Fixed azimuth angle (deg) for --mode vertical-scan")
    p.add_argument("--free-3d", action="store_true", help="Do not enforce slab normal in --mode atoms")

    p.add_argument("--width", type=float, default=9.0, help="Slice horizontal width in Angstrom")
    p.add_argument("--height", type=float, default=9.0, help="Slice vertical height in Angstrom")
    p.add_argument("--slab-tol", type=float, default=0.65, help="Max |d_perp| (A) to overlay atoms & bonds")
    p.add_argument("--grid-res", type=int, default=320, help="2D interpolation grid resolution per axis")
    p.add_argument("--bonds", default="auto", help="'auto' or comma-separated element pairs e.g. 'Fe-S,S-S'")
    p.add_argument("--vlim", type=float, default=None, help="Manual colorbar max (or +/- max for diverging fields)")
    p.add_argument("--label-mode", default="clean", choices=["clean", "all", "none"], help="Atom labeling density")
    p.add_argument("--save-csv", action="store_true", help="Also export full 2D grid CSV for custom R/Origin plotting")
    p.add_argument("--inspect-only", action="store_true", help="Only inspect plane candidates, spin alignment & bond stats")
    p.add_argument("--out-dir", default="./vasp_slice_out", help="Output directory")
    p.add_argument("--prefix", default="slice_output", help="Output filename prefix")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    run_pipeline(parser.parse_args())
