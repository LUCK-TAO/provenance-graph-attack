"""Generate the paper figure for macro-action reward timing and credit assignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#1F2937",
    "muted": "#5F6B7A",
    "line": "#7B8794",
    "blue": "#DCEBFA",
    "blue_edge": "#3E78B2",
    "green": "#E2F3E8",
    "green_edge": "#3A8D5D",
    "orange": "#FDEBD2",
    "orange_edge": "#C77A17",
    "red": "#F9E1E1",
    "red_edge": "#B94A48",
    "purple": "#EAE3F7",
    "purple_edge": "#7655A6",
    "gray": "#F2F4F7",
    "gray_edge": "#8A94A3",
    "white": "#FFFFFF",
}


def choose_chinese_font() -> str:
    available = {
        item.name for item in font_manager.fontManager.ttflist
    }
    for name in (
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ):
        if name in available:
            return name
    return "DejaVu Sans"


def add_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    *,
    face,
    edge,
    fontsize=12,
    linewidth=1.5,
    radius=0.13,
    zorder=2,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.03,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.35,
        zorder=zorder + 1,
    )
    return patch


def add_arrow(
    ax,
    start,
    end,
    *,
    color=None,
    linewidth=1.5,
    style="-|>",
    connectionstyle="arc3,rad=0",
    linestyle="-",
    zorder=1,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=linewidth,
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def section_label(ax, x, y, number, text):
    ax.text(
        x,
        y,
        f"{number}  {text}",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["ink"],
    )


def build_figure():
    plt.rcParams.update(
        {
            "font.family": choose_chinese_font(),
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        9,
        9.65,
        "宏动作奖励计算与信用分配时机",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["ink"],
    )

    # 1. Macro execution timeline.
    section_label(ax, 0.35, 9.08, "①", "宏动作执行：一次策略决策，连续五个同语义基元编辑")
    add_box(
        ax, 0.4, 7.35, 1.65, 1.05,
        "宏状态\n$s_t$",
        face=COLORS["blue"], edge=COLORS["blue_edge"], fontsize=13,
    )
    add_box(
        ax, 2.55, 7.35, 2.05, 1.05,
        "语义 Actor\n$a_t\\sim\\pi_\\theta(a|s_t)$",
        face=COLORS["purple"], edge=COLORS["purple_edge"], fontsize=12,
    )
    add_arrow(ax, (2.05, 7.88), (2.55, 7.88))

    group = FancyBboxPatch(
        (5.05, 7.02),
        6.0,
        1.72,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=1.3,
        edgecolor=COLORS["blue_edge"],
        facecolor="#F7FAFD",
        linestyle="--",
    )
    ax.add_patch(group)
    ax.text(
        8.05,
        8.52,
        "宏动作 $M_t$：五条边共享同一语义动作 $a_t$",
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["ink"],
    )
    edit_x = [5.32, 6.42, 7.52, 8.62, 9.72]
    for index, x in enumerate(edit_x, start=1):
        add_box(
            ax, x, 7.35, 0.86, 0.78,
            f"$e_{{t,{index}}}$\n$a_t$",
            face=COLORS["blue"], edge=COLORS["blue_edge"], fontsize=11,
            radius=0.08,
        )
        if index < 5:
            add_arrow(
                ax,
                (x + 0.86, 7.74),
                (edit_x[index], 7.74),
                linewidth=1.1,
            )
    add_arrow(ax, (4.6, 7.88), (5.05, 7.88))

    add_box(
        ax, 11.55, 7.35, 2.15, 1.05,
        "查询受害检测器\n仅在第5条边之后",
        face=COLORS["orange"], edge=COLORS["orange_edge"], fontsize=12,
    )
    add_arrow(ax, (11.05, 7.88), (11.55, 7.88))
    add_box(
        ax, 14.2, 7.35, 1.55, 1.05,
        "新状态\n$s_{t+1}$",
        face=COLORS["blue"], edge=COLORS["blue_edge"], fontsize=13,
    )
    add_arrow(ax, (13.7, 7.88), (14.2, 7.88))
    add_box(
        ax, 16.25, 7.35, 1.35, 1.05,
        "终止判断\n$d_t$",
        face=COLORS["gray"], edge=COLORS["gray_edge"], fontsize=12,
    )
    add_arrow(ax, (15.75, 7.88), (16.25, 7.88))

    ax.text(
        8.05,
        6.78,
        "不进行逐边查询；宏奖励不拆分为 $r_t/5$，五个基元编辑共同获得一次联合反馈",
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["red_edge"],
    )

    # 2. Reward computation.
    section_label(ax, 0.35, 6.18, "②", "奖励计算：只使用检测器可观察分数、固定阈值与编辑数量")
    add_arrow(
        ax,
        (12.62, 7.35),
        (12.62, 5.72),
        color=COLORS["orange_edge"],
        connectionstyle="arc3,rad=0.04",
    )
    ax.text(
        12.88,
        6.48,
        "$s_t(v),\\ s_{t+1}(v),\\ \\tau$",
        ha="left",
        va="center",
        fontsize=11,
        color=COLORS["muted"],
    )

    add_box(
        ax, 0.45, 4.35, 3.55, 1.12,
        "分数变化奖励 / 惩罚\n"
        "$r_{score}=\\mathrm{clip}\\!\\left(\\dfrac{s_t(v)-s_{t+1}(v)}"
        "{\\max(|s_t(v)|,|\\tau|,\\epsilon)},-1,1\\right)$",
        face=COLORS["green"], edge=COLORS["green_edge"], fontsize=10.5,
    )
    add_box(
        ax, 4.25, 4.35, 3.0, 1.12,
        "跨阈值奖励\n"
        "$r_{cross}=\\mathbb{I}[s_t(v)\\geq\\tau>s_{t+1}(v)]$",
        face=COLORS["green"], edge=COLORS["green_edge"], fontsize=10.5,
    )
    add_box(
        ax, 7.5, 4.25, 4.35, 1.32,
        "无标签间接影响\n"
        "$r_{ind}=0.5\\,\\mathrm{mean}(\\mathrm{Top5}(\\Delta s_i^+))$\n"
        "$\\quad+0.5\\tanh(N_{ind\\_cross}/2)$",
        face=COLORS["green"], edge=COLORS["green_edge"], fontsize=10.3,
    )
    add_box(
        ax, 12.0, 4.35, 2.25, 1.12,
        "编辑成本惩罚\n"
        "$r_{edit}=\\Delta E_t/L$",
        face=COLORS["red"], edge=COLORS["red_edge"], fontsize=10.5,
    )
    add_box(
        ax, 14.5, 4.18, 3.0, 1.46,
        "宏动作总奖励\n"
        "$r_t=1.0r_{score}$\n"
        "$+0.5r_{cross}+0.2r_{ind}-0.05r_{edit}$",
        face=COLORS["orange"], edge=COLORS["orange_edge"], fontsize=10.8,
        linewidth=1.8,
    )
    add_arrow(ax, (4.0, 4.91), (14.5, 4.91), linewidth=1.0)
    add_arrow(ax, (7.25, 4.91), (14.5, 4.91), linewidth=1.0)
    add_arrow(ax, (11.85, 4.91), (14.5, 4.91), linewidth=1.0)
    add_arrow(ax, (14.25, 4.91), (14.5, 4.91), linewidth=1.0)

    ax.text(
        9.0,
        3.87,
        "在线奖励不读取 metadata 真值；间接项只观察未被直接攻击的初始告警",
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    # 3. Transition, rollout, and GAE.
    section_label(ax, 0.35, 3.28, "③", "信用分配：宏内联合归因，宏间通过GAE传播，目标间相互隔离")
    add_box(
        ax, 0.65, 1.55, 4.55, 1.12,
        "一个宏 transition\n"
        "$(s_t,\\ a_t,\\ r_t,\\ s_{t+1},\\ d_t)$",
        face=COLORS["blue"], edge=COLORS["blue_edge"], fontsize=12.5,
    )
    add_arrow(
        ax,
        (15.9, 4.18),
        (4.55, 2.67),
        color=COLORS["orange_edge"],
        connectionstyle="arc3,rad=-0.13",
    )
    add_box(
        ax, 5.8, 1.55, 3.05, 1.12,
        "宏 rollout\n累计8个transition",
        face=COLORS["gray"], edge=COLORS["gray_edge"], fontsize=12,
    )
    add_arrow(ax, (5.2, 2.11), (5.8, 2.11))
    add_box(
        ax, 9.45, 1.35, 4.15, 1.52,
        "GAE优势估计\n"
        "$\\delta_t=r_t+\\gamma V(s_{t+1})(1-d_t)-V(s_t)$\n"
        "$A_t=\\delta_t+\\gamma\\lambda(1-d_t)A_{t+1}$",
        face=COLORS["purple"], edge=COLORS["purple_edge"], fontsize=11.5,
    )
    add_arrow(ax, (8.85, 2.11), (9.45, 2.11))
    add_box(
        ax, 14.2, 1.55, 3.2, 1.12,
        "PPO更新\nActor + Critic",
        face=COLORS["green"], edge=COLORS["green_edge"], fontsize=12.5,
    )
    add_arrow(ax, (13.6, 2.11), (14.2, 2.11))

    ax.text(
        11.52,
        0.92,
        "同一目标：后续宏奖励以 $\\gamma\\lambda$ 折扣回传   ｜   "
        "目标边界：$d_t=1$，奖励不传播到下一目标",
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["ink"],
    )
    ax.text(
        0.65,
        0.72,
        "PPO只更新语义动作 $a_t$ 的概率；五个候选端点由确定性合法选择器生成，不单独接收梯度。",
        ha="left",
        va="center",
        fontsize=11,
        color=COLORS["muted"],
    )

    fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.035)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_figures",
    )
    parser.add_argument(
        "--basename",
        default="fig_reward_timing_credit_assignment",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure = build_figure()
    outputs = {
        "png": args.output_dir / f"{args.basename}.png",
        "pdf": args.output_dir / f"{args.basename}.pdf",
        "svg": args.output_dir / f"{args.basename}.svg",
    }
    figure.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    figure.savefig(outputs["pdf"], bbox_inches="tight")
    figure.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(figure)
    for kind, path in outputs.items():
        print(f"{kind}: {path.resolve()}")


if __name__ == "__main__":
    main()
