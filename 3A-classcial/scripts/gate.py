#!/usr/bin/env python3
"""自评门禁：扫违禁词 + 跑配套代码 + 基础结构检查。
用法:
    python scripts/gate.py 1        # 校验系列 1
    python scripts/gate.py all      # 校验全部
未通过则以非零退出码结束，并打印原因。
"""
import re
import sys
import io
import subprocess
from pathlib import Path

# Windows 控制台默认 GBK，无法输出 emoji；强制 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
POSTS = ROOT / "posts"
CODE = ROOT / "code"

BANNED = ["显然", "易得", "不难看出", "众所周知", "obviously", "trivially"]
REQUIRED_SECTIONS = ["TL;DR", "工程痛点", "直觉", "数学推导", "代码实战", "工程踩坑", "小结"]


def check_post(md: Path) -> list[str]:
    """返回问题列表，空列表表示通过。"""
    errs = []
    text = md.read_text(encoding="utf-8")

    # 1. 违禁词
    for w in BANNED:
        for m in re.finditer(re.escape(w), text):
            line = text[: m.start()].count("\n") + 1
            errs.append(f"[违禁词] '{w}' @ {md.name}:{line}")

    # 2. 必备结构
    for sec in REQUIRED_SECTIONS:
        if sec not in text:
            errs.append(f"[缺结构] 未找到章节标记 '{sec}' in {md.name}")

    # 3. 比喻 + 面试追问
    if "🔥" not in text:
        errs.append(f"[缺面试追问] 未找到 🔥 标记 in {md.name}")
    if "⭐" not in text:
        errs.append(f"[缺结论块] 未找到 ⭐ 标记 in {md.name}")

    # 4. 博客自包含：不得有指向 code/ 的死链，且须内嵌完整代码块
    for m in re.finditer(r"\.\./code/", text):
        line = text[: m.start()].count("\n") + 1
        errs.append(f"[死链] 指向 ../code/ 的相对链接上传博客后会失效 @ {md.name}:{line}")
    if "AUTO-EMBED:BEGIN" not in text:
        errs.append(f"[缺完整代码] 未内嵌完整可跑代码块 in {md.name}（跑 scripts/embed_code.py）")

    # 5. 向量符号：组合箭头 U+20D7 在多数博客字体下显示成 □?，禁止出现
    for m in re.finditer("⃗", text):
        line = text[: m.start()].count("\n") + 1
        errs.append(f"[渲染坑] U+20D7 组合箭头会显示成方框问号 @ {md.name}:{line}（跑 scripts/fix_math.py）")

    return errs


def run_code(py: Path) -> list[str]:
    """实际执行配套代码，非零退出即失败。"""
    if not py.exists():
        return [f"[缺代码] {py.name} 不存在"]
    r = subprocess.run(
        [sys.executable, str(py)],
        capture_output=True, text=True, cwd=str(ROOT),
        env={"MPLBACKEND": "Agg", **_env()},
    )
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-15:]
        return [f"[代码报错] {py.name} 退出码 {r.returncode}:\n    " + "\n    ".join(tail)]
    return []


def _env():
    import os
    return dict(os.environ)


def gate(series: str) -> list[str]:
    errs = []
    md = POSTS / f"series-{series}.md"
    py = CODE / f"series-{series}.py"
    if not md.exists():
        return [f"[缺正文] {md.name} 不存在"]
    errs += check_post(md)
    errs += run_code(py)
    return errs


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/gate.py <series|all>")
        sys.exit(2)
    arg = sys.argv[1]
    targets = ["1", "2A", "2B", "3A", "3B", "4", "5", "6"] if arg == "all" else [arg]

    all_ok = True
    for s in targets:
        errs = gate(s)
        if errs:
            all_ok = False
            print(f"❌ 系列 {s} 未通过：")
            for e in errs:
                print("   " + e)
        else:
            print(f"✅ 系列 {s} 全绿")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
