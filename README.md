# VASP Volumetric Slice & Bond Quantifier (`vasp-volumetric-slice`)

一个轻量、纯 Python 实现的 VASP 三维体数据后处理工具（兼顾 Agent Skill 与独立命令行运行），可直接从 **VASP 原始输出文件**出发，自动完成**自旋通道对齐、周期性最优切面选择、真实 3D 化学键定量提取，以及无“AI 味”的出版级矢量出图**。

---

## 1. 环境依赖（Requirements）

仅需标准科学计算 Python 栈（无需安装 VESTA、vaspkit 或 R）：

```bash
pip install -r requirements.txt
```

依赖清单（见 [`requirements.txt`](requirements.txt)）：
- `numpy >= 1.23.0`
- `scipy >= 1.9.0`
- `pandas >= 1.5.0`
- `matplotlib >= 3.6.0`

---

## 2. 适用范围与边界说明（Scope & Limitations）

### 适用范围
1. **可直接读取的 VASP 输出文件**：
   - `ELFCAR`：自动识别单/双自旋（`ISPIN=2` 时直接解析文件内的 Block 1 和 Block 2，无需再用 vaspkit 拆分）。
   - `CHGCAR` / `CHG` / `PARCHG`：自动跳过 PAW augmentation block，按晶胞体积归一化为 $e/\text{Å}^3$；`ISPIN=2` 时自动提取总电荷密度（Block 1）与自旋密度（Block 2）。
   - **差分电荷密度（CDD）**：支持直接传入 3 个原始 `CHGCAR`（`--cdd-ab`、`--cdd-a`、`--cdd-b`），脚本自动做 $\Delta\rho = \rho_{\text{AB}} - \rho_{\text{A}} - \rho_{\text{B}}$ 并切面出图。
   - `LOCPOT`：局域静电势（eV），用于分析活性位点上方局域电势分布。
   - `OUTCAR`（可选，同目录自动识别）：自动读取最终 `magnetization (x)` 原子磁矩，判断 `spin-up / spin-down` 谁是局域 `majority / minority`，防止跨体系自旋通道画反。
2. **适用的体系类型（不局限于单原子催化剂）**：
   - **三维周期性体相晶体与矿物**（如**黄铁矿 $\text{FeS}_2$**、氧化物、硫化物、MOF、合金）：支持任意晶系（立方、六方、单斜、三斜），支持密勒指数切面 `(hkl)`，支持跨晶胞周期性边界（MIC）的化学键自动展开与无缝投影。
   - **表面平板模型与单原子体系（Slabs / SACs）**：自动识别真空层方向，支持垂直方位角扫描（`vertical-scan`）对齐多个对比体系。
   - **孤立分子 / 团簇模型**：支持三原子精确共面与多原子 SVD 最小二乘平面拟合。

### 不适用范围与注意事项
1. **原子核内区 PAW 伪电荷限制**：VASP `CHGCAR/ELFCAR` 在原子核近心区为 PAW 赝势密度，脚本已自动避开两端核区、在 `[0.20d, 0.80d]` 区间提取键鞍点与中点值，**不适合直接用于核芯区全电子拓扑（QTAIM 核尖峰）分析**。
2. **CDD 三文件网格必须一致**：使用 `--cdd-ab/a/b` 直接相减时，三个 `CHGCAR` 必须具有完全一致的晶格矢量与 `NGXF × NGYF × NGZF` 网格。
3. **非共线磁矩（`LNONCOLLINEAR=.TRUE.`）**：不支持 4 分量旋量网格作为共线 `majority/minority` 解析。
4. **缺少 `OUTCAR` 时的自旋翻转**：若单独复制 `ELFCAR` 而未带 `OUTCAR`，且对比体系发生局域磁矩方向翻转，默认假设 Block 1 为 majority，建议把 `OUTCAR` 一并放在同目录或检查 `--inspect-only` 输出。

---

## 3. 典型使用示例（可直接复制运行）

### 示例 A：黄铁矿 $\text{FeS}_2$ 等体相晶体（非单原子体系）
1. **自动诊断候选切面与化学键（强烈建议先跑一次预览信息）**：
   ```bash
   python scripts/analyze_and_plot.py --input ./FeS2_bulk/ELFCAR --mode auto --inspect-only
   ```
2. **按晶体学 `(110)` 面过 `Fe1` 原子切出（同时无缝铺开周期性 $\text{FeS}_6$ 与 $\text{S–S}$ 键网络）**：
   ```bash
   python scripts/analyze_and_plot.py \
     --input ./FeS2_bulk/ELFCAR \
     --names "Pyrite FeS2 (110)" \
     --mode hkl --hkl 1 1 0 --anchor Fe1 \
     --width 10 --height 10 \
     --bonds "Fe-S,S-S" \
     --out-dir ./out_pyrite --prefix FeS2_110_ELF
   ```
3. **按特定 $\text{Fe–S–S}$ 二聚体三原子平面精确切出（自动处理跨边界周期键）**：
   ```bash
   python scripts/analyze_and_plot.py \
     --input ./FeS2_bulk/ELFCAR \
     --mode atoms --atoms Fe1,S1,S2 \
     --width 8 --height 8 \
     --out-dir ./out_pyrite --prefix FeS2_FeSS_ELF
   ```

### 示例 B：单原子 / 表面吸附体系（双体系对比 + 自旋分辨面板）
```bash
# 1) 主图：双体系平均 ELF 对比（当两体系原子总数/编号不同时，推荐 --mode auto 自动对齐活性中心）
python scripts/analyze_and_plot.py \
  --input ./FeN4_ClO2/ELFCAR ./FeN4_axCl_ClO2/ELFCAR \
  --names "FeN4/NC" "Cl-FeN4/NC" \
  --mode auto \
  --width 8.5 --height 9.0 \
  --out-dir ./out_sac --prefix Fig_ELF_Main

# 2) SI 图：自旋分辨 2x3 面板 (Majority / Minority / Maj-Min)
python scripts/analyze_and_plot.py \
  --input ./FeN4_ClO2/ELFCAR ./FeN4_axCl_ClO2/ELFCAR \
  --names "FeN4/NC" "Cl-FeN4/NC" \
  --mode auto \
  --spin panel --vlim 0.42 \
  --width 8.5 --height 9.0 \
  --out-dir ./out_sac --prefix Fig_ELF_Spin
```

### 示例 C：直接从 3 个 VASP `CHGCAR` 计算并绘制差分电荷密度（CDD）2D 剖面
```bash
python scripts/analyze_and_plot.py \
  --cdd-ab ./ads/CHGCAR --cdd-a ./slab/CHGCAR --cdd-b ./mol/CHGCAR \
  --names "Cl-FeN4 / ClO2-" \
  --mode vertical-scan --anchor Fe1 --atoms Fe1,O1,Cl1,O2 \
  --out-dir ./out_cdd --prefix Fig_CDD_2D
```

### 示例 D：提取自旋密度（`CHGCAR`）或局域静电势（`LOCPOT`）
```bash
# 自旋密度剖面 (直接从 ISPIN=2 的 CHGCAR 提取 Block 2)
python scripts/analyze_and_plot.py --input ./FeS2_slab/CHGCAR --field spin --mode hkl --hkl 1 0 0 --anchor Fe1

# 局域静电势剖面
python scripts/analyze_and_plot.py --input ./FeS2_slab/LOCPOT --field locpot --mode vertical-scan --anchor Fe1 --atoms Fe1,S1,O1
```

---

## 4. 输出文件说明

每次运行会在 `--out-dir` 生成：
- `<prefix>.png`（600 dpi 高清位图）
- `<prefix>.pdf`（TrueType 字体，可在 Adobe Illustrator 中直接编辑文字与矢量框）
- `<prefix>.svg`（纯 `<text>` 可编辑矢量图）
- `<prefix>_atoms.csv`（切片窗口内所有投影原子及离面残差 `d_perp`）
- `<prefix>_bonds.csv`（真实 3D 键长、键中点值、核外鞍点极小值 `saddle_min`、成键极大值 `basin_max`）
- `<prefix>_diagnostics.json`（晶格维度判断、自旋对齐来源、切面基矢与候选平面排名）
- 添加 `--save-csv` 时额外导出 `<prefix>_slice.csv`（供 R/Origin 二次自定义绘图）。

---

## 5. 分享给他人的方式

只需将整个 `vasp-volumetric-slice/` 文件夹打包发给对方：
- **作为独立 Python 工具使用**：`pip install -r requirements.txt` 后直接运行 `python scripts/analyze_and_plot.py ...`。
- **作为 AI Agent Skill 使用**：将该文件夹放入项目 `.agents/skills/` 或个人全局 skills 目录即可，AI 会自动遵循 `SKILL.md` 的标准化流程执行。
