---
name: vasp-volumetric-slice
description: Automatically inspect, slice, quantify 3D chemical bonds, and generate journal-standard 2D figures directly from VASP volumetric output files (ELFCAR, CHGCAR, LOCPOT, PARCHG, OUTCAR) for bulk crystals (e.g., pyrite FeS2), slabs/interfaces, single-atom catalysts, and molecules.
---

# VASP Volumetric Slice & Bond Quantifier

This skill converts raw VASP 3D volumetric grids (`ELFCAR`, `CHGCAR`, `CHG`, `LOCPOT`, `PARCHG`, optionally `OUTCAR`) into publication-ready 2D cross-section figures and quantitative 3D bond statistics without manual VESTA/vaspkit slicing.

## 1. Scope & Applicability (适用范围)

### Supported Inputs (直接支持从 VASP 输出开始)
- **Volumetric files**: `ELFCAR` (`ISPIN=1/2` dual-block auto-parsed), `CHGCAR` / `CHG` (total charge density + spin density block, automatically skips PAW augmentation occupancies), `LOCPOT` (electrostatic potential in eV), `PARCHG`, or precomputed `CHGDIFF`.
- **Direct CDD subtraction**: `--cdd-ab CHGCAR_AB --cdd-a CHGCAR_A --cdd-b CHGCAR_B` computes $\Delta\rho = \rho_{\text{AB}} - \rho_{\text{A}} - \rho_{\text{B}}$ directly.
- **Automatic spin alignment**: If `OUTCAR` is in the same directory as the grid file, local magnetic moments (`magnetization (x)`) are parsed automatically to map VASP `spin-up / spin-down` to physical `majority / minority`.

### Supported Systems (体系适用范围)
- **Periodic bulk crystals & minerals** (e.g., pyrite $\text{FeS}_2$, $\text{Fe}_2\text{O}_3$, $\text{TiO}_2$, alloys, MOFs): handles arbitrary lattice shapes (cubic, hexagonal, monoclinic, triclinic), Minimum Image Convention (MIC) bond unwrapping across cell boundaries, periodic atom/bond tiling inside the slice window, and `(hkl)` Miller planes.
- **Slabs, interfaces & single-atom catalysts (SACs)**: detects vacuum axis automatically, runs vertical azimuth scanning (`--mode vertical-scan`) around an active site to align multiple systems into the same $(u, v)$ frame.
- **Molecular clusters in periodic boxes**: supports 3-atom exact planes and $\ge 4$-atom SVD best-fit planes (`--mode atoms`).

### Out-of-Scope / Cautions (不适用范围与注意事项)
- **Core-region PAW inaccuracy**: VASP `CHGCAR`/`ELFCAR` contain pseudo-valence grids; near atomic nuclei ($<0.5\ \text{Å}$), values reflect PAW pseudo-density rather than all-electron core density (`AECCAR0/2`). Bond midpoints and saddle points in `[0.20d, 0.80d]` are safe; do not interpret raw core peaks inside nuclei.
- **Grid mismatch in CDD**: Direct subtraction (`--cdd-ab/a/b`) requires identical lattice vectors and `NGXF/NGYF/NGZF` grid dimensions across all three `CHGCAR`s.
- **Non-collinear magnetism (`LNONCOLLINEAR=.TRUE.`)**: 4-component spinor vector grids are not treated as collinear `majority/minority`.
- **Spin flip without `OUTCAR`**: `ELFCAR` alone does not store local magnetic moment signs; if comparing two spin-polarized `ELFCAR` files without `OUTCAR`, verify majority/minority assignment manually or supply `OUTCAR`.

---

## 2. Standard Execution Workflow (Agent SOP)

### Step 1: Inspect Geometry, Spin Alignment & Candidate Planes
Always run `--inspect-only` first before plotting:
```bash
python "<skill_dir>/scripts/analyze_and_plot.py" \
  --input <path_to_ELFCAR_or_vasp_dir> \
  --mode auto \
  --inspect-only \
  --out-dir <out_dir>
```
Check the console output and `<prefix>_diagnostics.json`:
1. Confirm `dimensionality` (`bulk` vs `slab` vs `molecule`).
2. Confirm `spin_alignment` (check if `OUTCAR` was found and which block is `majority`).
3. Review `candidate_planes` and `target_residuals_A` (`d_perp`).

### Step 2: Choose the Physically Optimal Plane Mode
- **For bulk crystals / minerals (e.g., pyrite $\text{FeS}_2$)**:
  - Use `--mode hkl --hkl 1 0 0 --anchor Fe1` (or `(110)`, `(111)`) for standard crystallographic planes, **or**
  - Use `--mode atoms --atoms Fe1,S1,S2` (where `S1-S2` is a disulfide dimer) to slice the exact 3D $\text{Fe–S–S}$ chemical bonding plane across periodic boundaries.
- **For slabs / surfaces / adsorbates**:
  - Use `--mode vertical-scan --anchor Fe1 --atoms Fe1,O1,Cl1,O2` so the surface normal stays vertical and the horizontal azimuth minimizes out-of-plane residuals across all systems consistently.
- **For multi-system comparison**:
  - Pass multiple `--input` paths and `--names` together so shared colorbar limits (`vmin/vmax`) and identical panel sizing are enforced automatically.

### Step 3: Render Publication-Grade Figures & Bond Tables
```bash
python "<skill_dir>/scripts/analyze_and_plot.py" \
  --input <sys1> <sys2> \
  --names "System A" "System B" \
  --mode <chosen_mode> \
  --out-dir <out_dir> \
  --prefix <figure_name>
```
For spin-resolved `Maj / Min / Δ(Maj−Min)` comparison panels (when `ISPIN=2`), add `--spin panel`.

---

## 3. Journal-Standard Anti-AI-Flavor Rules (出图规范)

When presenting or customizing outputs from this skill, strictly adhere to these scientific visualization rules:
1. **Zero internal text clutter**: Never place floating narrative badges, explanatory sentences, or decorative callout pills inside the heatmap area (e.g., avoid `"No ligand here"` boxes). Put concise identifiers in the panel title `(a) FeS2 (110)` or figure caption.
2. **True physical aspect ratio**: Keep `aspect='equal'` (1:1 Å scale) with clean `In-plane distance (Å)` and `Vertical distance (Å)` axes.
3. **Standard colormaps & symmetric zeroing**:
   - `ELF`: `0.0–1.0` VESTA rainbow (`#00007F` → `#00FF80` → `#D90000`).
   - `ΔELF / CDD / Spin density`: symmetric diverging palette (`blue–white–red`) strictly centered at `0` (`[-vlim, +vlim]`).
4. **Restrained structural overlay**: Use small crisp white circles with dark borders (`1 pt`) and subtle dashed bond segments only for coplanar/near-slice bonds (`|d_perp| <= slab_tol`). In dense bulk crystals (e.g., pyrite), keep `--label-mode clean` or `none` to avoid overlapping atom labels.
5. **Vector editability**: Always export `PNG (600 dpi)` alongside `PDF` (`fonttype=42`) and `SVG` (`svg.fonttype='none'`) so text remains editable in Adobe Illustrator / Inkscape.
