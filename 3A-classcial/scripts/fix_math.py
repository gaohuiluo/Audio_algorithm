#!/usr/bin/env python3
r"""修复向量符号渲染问题：组合箭头 U+20D7 (x⃗) 在多数博客字体下显示成 □?。

策略（平台支持 MathJax/KaTeX）：
- 代码围栏(``` fences)之外的行内代码 `..math..` 与散文里的数学 → 转成 $...$ LaTeX
  （\vec{}、\|、^{\top} 等，箭头由 \vec 正确渲染）。
- 代码围栏之内(python/公式块，MathJax 不渲染) → 只把异体 unicode ASCII 化
  （箭头去掉、ᵀ→^T、ᴴ→^H、⁻¹→^-1、‖→||），保持等宽可读、不出方框。
用法: python scripts/fix_math.py [文件...]   默认处理 posts/*.md 与 code/*.py
"""
import re
import sys
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARROW = "\u20d7"

# 非围栏(LaTeX)环境下的 unicode → LaTeX 映射（不含箭头，箭头单独处理）
LATEX_MAP = {
    "ᴴ": r"^{\mathsf{H}}", "ᵀ": r"^{\top}", "⁻": "^", "¹": "1", "²": "^{2}", "³": "^{3}",
    "‖": r"\|", "·": r"\cdot ", "⊙": r"\odot ", "∝": r"\propto ", "∇": r"\nabla ",
    "≈": r"\approx ", "−": "-", "←": r"\leftarrow ", "…": r"\dots ", "√": r"\sqrt",
    "θ": r"\theta ", "μ": r"\mu ", "λ": r"\lambda ", "ε": r"\varepsilon ",
    "ω": r"\omega ", "τ": r"\tau ", "δ": r"\delta ", "γ": r"\gamma ", "ξ": r"\xi ",
    "∈": r"\in ", "∞": r"\infty ", "≤": r"\le ", "≥": r"\ge ", "≠": r"\ne ",
}

# 围栏(等宽代码/公式)环境下的 ASCII 化映射（保留能正常显示的希腊字母/·/²）
ASCII_MAP = {
    ARROW: "", "ᴴ": "^H", "ᵀ": "^T", "⁻¹": "^-1", "⁻": "^-", "‖": "||",
}


def to_latex(s: str) -> str:
    """把一段裸数学字符串转成 LaTeX（不含首尾 $）。"""
    s = re.sub(r"(.)" + ARROW, r"\\vec{\1}", s)          # x⃗ -> \vec{x}
    s = re.sub(r"_([A-Za-z0-9]{2,})", r"_{\1}", s)        # 多字符下标 _opt -> _{opt}
    for k, v in LATEX_MAP.items():
        s = s.replace(k, v)
    return re.sub(r"\s{2,}", " ", s).strip()


def asciify(s: str) -> str:
    for k, v in ASCII_MAP.items():
        s = s.replace(k, v)
    return s


def fix_markdown(text: str) -> str:
    out, i, n = [], 0, len(text)
    fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fence = not fence
            out.append(line)
            continue
        if fence:
            out.append(asciify(line))
            continue
        # 非围栏：1) 行内代码里含箭头的 -> $...$ ; 2) 散文里的裸箭头 token -> $...$
        def repl_code(m):
            inner = m.group(1)
            return "$" + to_latex(inner) + "$" if ARROW in inner else m.group(0)
        line = re.sub(r"`([^`\n]*)`", repl_code, line)
        # 裸箭头：字母+箭头，可带 (..) 参数或下标
        line = re.sub(r"[A-Za-z]" + ARROW + r"(?:\([^)]*\)|_\{[^}]*\}|_\w|ᴴ|ᵀ)*",
                      lambda m: "$" + to_latex(m.group(0)) + "$", line)
        out.append(line)
    return "".join(out)


def main():
    targets = sys.argv[1:] or (glob.glob(str(ROOT / "posts" / "*.md")) +
                               glob.glob(str(ROOT / "code" / "*.py")))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for f in targets:
        p = Path(f)
        t = p.read_text(encoding="utf-8")
        before = t.count(ARROW)
        if p.suffix == ".md":
            t = fix_markdown(t)
        else:  # .py：仅注释含 unicode，整体 ASCII 化安全
            t = asciify(t)
        p.write_text(t, encoding="utf-8")
        print(f"{p.name}: {before} 处箭头 -> 剩 {t.count(ARROW)}")


if __name__ == "__main__":
    main()
