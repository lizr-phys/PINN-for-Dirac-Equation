# pinn_harmonic_oscillator_report - Design Spec

> Human-readable design narrative - rationale, audience, style, color choices, content outline. Read once by downstream roles for context.
>
> Machine-readable execution contract: `spec_lock.md`. Executor re-reads `spec_lock.md` before every SVG page.

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | pinn_harmonic_oscillator_report |
| **Canvas Format** | PPT 16:9 (1280x720) |
| **Page Count** | 12 |
| **Design Style** | instructional + swiss-minimal |
| **Target Audience** | 课题组/课程汇报听众：了解量子力学与基础数值计算，但不一定熟悉 PINN 程序实现 |
| **Use Case** | 学术汇报：解释一维线性谐振子 PINN 程序做了什么、原理、改进流程，以及对后续 Dirac 方程问题的启示 |
| **Created Date** | 2026-06-19 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280x720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 60px, top 54px, bottom 44px |
| **Content Area** | 1160x622 |

---

## III. Visual Theme

### Theme Style

- **Mode**: instructional
- **Visual style**: swiss-minimal
- **Theme**: Light theme
- **Tone**: 克制、学术、清晰、程序导向。以结构解释为主，不做炫技视觉。

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FAFBFC` | Page background |
| **Secondary bg** | `#EEF3F8` | Subtle panels, equation bands, figure beds |
| **Primary** | `#1C4E80` | Titles, main rules, core process arrows |
| **Accent** | `#D88C2D` | Key loss terms, error callouts, current-step emphasis |
| **Secondary accent** | `#4FA3A5` | Data anchors, transfer/Dirac route |
| **Body text** | `#1F2933` | Main body text |
| **Secondary text** | `#52606D` | Captions, source notes |
| **Tertiary text** | `#7B8794` | Footers, page numbers |
| **Border/divider** | `#CBD5E1` | Hairlines and figure frames |
| **Grid** | `#E2E8F0` | Light chart/grid guide |
| **Surface** | `#FFFFFF` | Cards and tables |
| **Success** | `#2F7D59` | Confirmed result markers |
| **Warning** | `#B45309` | Limitations / caution |

### Gradient Scheme

No decorative gradients. Only use flat fills and single-weight rules, consistent with swiss-minimal.

---

## IV. Typography System

### Font Plan

**Typography direction**: safe single sans, weight contrast instead of family contrast.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | Microsoft YaHei | Arial | sans-serif |
| **Body** | Microsoft YaHei | Arial | sans-serif |
| **Emphasis** | Microsoft YaHei | Arial | sans-serif |
| **Code** | - | Consolas, Courier New | monospace |

**Per-role font stacks**

- Title: `"Microsoft YaHei", Arial, sans-serif`
- Body: `"Microsoft YaHei", Arial, sans-serif`
- Emphasis: `"Microsoft YaHei", Arial, sans-serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 20px.

| Purpose | Ratio to body | Current Project | Weight |
| ------- | ------------- | --------------- | ------ |
| Cover title | 2.8-3.4x | 64px | Bold |
| Chapter / section opener | 2-2.4x | 44px | Bold |
| Page title | 1.6-1.8x | 34px | Bold |
| Hero number | 2.4-3x | 52px | Bold |
| Subtitle | 1.2-1.4x | 24px | SemiBold |
| **Body content** | **1x** | **20px** | Regular |
| Annotation / caption | 0.7-0.8x | 14-16px | Regular |
| Page number / footnote | 0.55-0.65x | 12px | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area**: 54px top margin; page title left-aligned, section label or page number on right.
- **Content area**: 1160x590, organized on a 12-column implied grid.
- **Footer area**: small source note or page number, no large footer band.

### Layout Pattern Library

| Pattern | Use in this deck |
| ------- | ---------------- |
| Single column centered | Cover and closing page |
| Asymmetric split (3:7 / 4:8) | Formula + explanation, paper/web understanding |
| Three/four column cards | Loss terms, improvement steps, Dirac implications |
| Matrix grid | Program components and comparison summaries |
| Center-radiating | Orthogonality principle diagram |
| Negative-space-driven | Main takeaway pages |
| Figure with native annotation layer | Result and paper-architecture pages |

### Spacing Specification

**Universal**

| Element | Recommended Range | Current Project |
| ------- | ---------------- | --------------- |
| Safe margin from canvas edge | 40-60px | 60px |
| Content block gap | 24-40px | 32px |
| Icon-text gap | 8-16px | 12px |

**Card-based layouts**

| Element | Recommended Range | Current Project |
| ------- | ---------------- | --------------- |
| Card gap | 20-32px | 24px |
| Card padding | 20-32px | 22px |
| Card border radius | 0-8px | 6px maximum |
| Single-row card height | 530-600px | 540px |
| Double-row card height | 245-285px each | 250px |
| Three-column card width | 350-370px each | 360px |

**Non-card containers**

- Use whitespace, rules, and aligned text blocks rather than nested cards.
- Equations and figures sit on subtle `secondary_bg` bands only when they need separation.
- Lines and arrows use one consistent stroke family.

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `tabler-outline`
- **Stroke width**: 2
- **Usage method**: SVG placeholder `<use data-icon="tabler-outline/icon-name" .../>`

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Quantum / oscillator | `tabler-outline/atom` | P01, P03 |
| Program code | `tabler-outline/code` | P04 |
| Function / formula | `tabler-outline/math-function` | P03, P05 |
| Training route | `tabler-outline/route` | P04, P07 |
| Results chart | `tabler-outline/chart-bar` | P09 |
| Data anchors | `tabler-outline/database` | P08 |
| Paper reading | `tabler-outline/book` | P10 |
| Objective | `tabler-outline/target` | P02 |
| Training signal | `tabler-outline/activity` | P06 |
| Implication | `tabler-outline/bulb` | P11 |
| Experiment / validation | `tabler-outline/flask` | P09 |
| Completed claim | `tabler-outline/circle-check` | P12 |
| Flow arrow | `tabler-outline/arrow-right` | P04, P07, P11 |
| Code settings | `tabler-outline/settings-code` | P07 |
| Network / multi-component | `tabler-outline/network` | P11 |

---

## VII. Visualization Reference List

This deck does not use chart-library SVG templates. All visualizations are custom native SVG diagrams or user-provided result images.

**Runners-up considered**

- `process_flow` | rejected: the program flow needs tighter custom alignment with code modules.
- `timeline_horizontal` | rejected: the improvement process is not chronological enough to justify a timeline template.
- `grouped_bar_chart` | rejected: result summary is already generated as a publication-style figure and should be preserved.

---

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | Reference | text_policy | page_role |
| -------- | ---------- | ----- | ------- | ---- | -------------- | ----------- | ------ | --------- | ----------- | --------- |
| formula_schrodinger.png | 709x121 | 5.86 | Block equation on P03 | Latex Formula | formula-block | formula | Rendered | Stationary oscillator Schrodinger equation | none | local |
| formula_boundary.png | 576x121 | 4.76 | Boundary parameterization on P05 | Latex Formula | formula-block | formula | Rendered | Boundary factor and symmetric network output | none | local |
| formula_loss.png | 1284x51 | 25.18 | Total loss formula on P06 | Latex Formula | formula-block | formula | Rendered | Total training objective | none | local |
| result_summary.png | 2409x1513 | 1.59 | Main validation evidence on P09 | Data figure | #19 Image floating in whitespace with thin frame and caption | user | Existing | Energy error, wavefunction error, data anchors, wavefunction curves | embedded | hero_figure |
| multi_wavefunction_comparison.png | 1600x1800 | 0.89 | Backup / detailed wavefunction evidence | Data figure | #19 Image floating in whitespace with thin frame and caption | user | Existing | Existing program output | embedded | local |
| energy_levels.png | 1400x900 | 1.56 | Backup energy-level comparison | Data figure | #19 Image floating in whitespace with thin frame and caption | user | Existing | Existing program output | embedded | local |
| paper_pinn_architecture.png | 771x319 | 2.42 | Paper-method source reference on P10 | Paper figure crop | #19 Image floating in whitespace with thin frame and caption | user | Existing | PINN eigenvalue architecture figure from paper-reader assets | embedded | local |
| loss_curve_n0.png | 1600x1000 | 1.60 | Backup training loss for n=0 | Data figure | #19 Image floating in whitespace with thin frame and caption | user | Existing | Existing program output | embedded | backup |
| loss_curve_n1.png | 1600x1000 | 1.60 | Backup training loss for n=1 | Data figure | #19 Image floating in whitespace with thin frame and caption | user | Existing | Existing program output | embedded | backup |
| loss_curve_n2.png | 1600x1000 | 1.60 | Backup training loss for n=2 | Data figure | #19 Image floating in whitespace with thin frame and caption | user | Existing | Existing program output | embedded | backup |

---

## IX. Content Outline

| Page | Rhythm | Title | Main message | Visual plan |
| ---- | ------ | ----- | ------------ | ----------- |
| P01 | anchor | 一维线性谐振子的 PINN 求解 | 汇报对象是当前程序，而不是泛泛介绍 PINN。 | Large title, compact subtitle, four-keyword baseline |
| P02 | breathing | 这段程序到底完成了什么 | 把谐振子本征方程转成可训练的物理约束问题，并求出前三个态。 | Four-part statement: object, method, result, outlook |
| P03 | dense | 物理问题先写清楚 | 可验证性来自解析能级、解析波函数和明确边界。 | Equation band + three facts |
| P04 | dense | 程序是一条从配点到结果的训练管线 | `model -> residual -> losses -> train -> evaluate -> export` 是核心骨架。 | Horizontal pipeline diagram |
| P05 | dense | 波函数不是直接输出，而是被物理先验包裹 | 边界因子和宇称嵌入把先验写进函数空间。 | Formula + boundary/parity schematic |
| P06 | dense | 损失函数把“像物理解”具体化 | PDE、归一化、正交、数据点和能量项分别解决不同问题。 | Loss equation + role matrix |
| P07 | dense | 改进不是一次重写，而是增量加约束 | 从基态程序扩展到多态、数据点和完整导出。 | Improvement ladder |
| P08 | dense | 五组数据点是锚点，不是替代物理方程 | 数据损失把少量可靠样本并入物理训练。 | Anchor-point schematic + practical meaning |
| P09 | dense | 结果说明程序学到了前三个本征态 | 能量误差小，高态波函数误差上升但仍保持形状。 | `result_summary.png` full figure with callouts |
| P10 | dense | 论文和网页给的是方法脉络 | 论文提供核心原则，网页帮助确认代码/方向，项目做了谐振子化取舍。 | Paper architecture image + interpretation cards |
| P11 | dense | 对 Dirac 方程的启示只在迁移逻辑 | 先可验证，再重写残差、边界和守恒约束。 | Current-to-Dirac transfer route |
| P12 | anchor | 当前结论与下一步 | 程序已完成多态谐振子验证，下一步是更稳健和更复杂方程迁移。 | Three conclusion blocks |

---

## X. Speaker Notes Plan

- Notes are Chinese, explanatory, and aligned with instructional mode.
- Each page starts by saying what question this slide answers.
- Avoid overclaiming: Dirac material is kept as a light outlook, not a completed result.
- Mention paper and webpage understanding briefly: they establish method context; the program design remains grounded in the oscillator equation.

---

## XI. Technical Constraints

- SVG viewBox must be `0 0 1280 720`.
- No `rgba()`, `<style>`, `class`, `<foreignObject>`, `<mask>`, scripts, or animations.
- Use inline SVG attributes only.
- All images are project-local under `images/`; formula PNGs and data figures should use `preserveAspectRatio="xMidYMid meet"`.
- One icon library only: `tabler-outline`, stroke width 2.
- Fonts must remain PPT-safe: Microsoft YaHei / Arial / Consolas.
