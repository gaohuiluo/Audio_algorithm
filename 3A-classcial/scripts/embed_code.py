#!/usr/bin/env python3
"""把 code/series-N.py 的完整源码内嵌进 posts/series-N.md，使文章自包含、可整篇上传博客。

做两件事（幂等，可反复运行）：
1) 把正文里指向 ../code/series-N.py 的外链句，替换成不带死链的表述；
2) 在「## 5.」工程踩坑章节之前，插入 "### 完整可跑代码" 小节，
   内嵌整份 .py（带 AUTO-EMBED 标记，重跑时先删旧的再插新的）。

用法: python scripts/embed_code.py            # 处理全部
      python scripts/embed_code.py 1 3A       # 只处理指定篇
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSTS = ROOT / "posts"
CODE = ROOT / "code"
SERIES = ["1", "2A", "2B", "3A", "3B", "4", "5", "6"]

BEGIN = "<!-- AUTO-EMBED:BEGIN 完整代码 由 scripts/embed_code.py 生成，勿手改 -->"
END = "<!-- AUTO-EMBED:END -->"


def build_block(series: str, code: str) -> str:
    return (
        f"{BEGIN}\n"
        f"### 完整可跑代码\n\n"
        f"> 以下为本篇配套的完整脚本，已实际执行通过。**直接复制保存为 `series-{series}.py`，"
        f"`python series-{series}.py` 即可复现上面所有配图**（需 `numpy` / `scipy` / `matplotlib`）。\n\n"
        f"```python\n{code.rstrip()}\n```\n"
        f"{END}"
    )


def dead_link_re(series: str):
    # 匹配任何提到 ../code/series-N.py 或 code/series-N.py 的整行
    return re.compile(rf"^.*code/series-{re.escape(series)}\.py.*$", re.MULTILINE)


def process(series: str) -> str:
    md_path = POSTS / f"series-{series}.md"
    py_path = CODE / f"series-{series}.py"
    if not md_path.exists() or not py_path.exists():
        return f"跳过 {series}：缺 md 或 py"

    text = md_path.read_text(encoding="utf-8")
    code = py_path.read_text(encoding="utf-8")

    # 1) 去掉旧的自动嵌入块（幂等）
    text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", text, flags=re.DOTALL).rstrip() + "\n"

    # 2) 把死链行改写为自包含表述（保留原句语义，去掉链接）
    def repl(m):
        line = m.group(0)
        # 顶部元信息行里的 "配套代码：[...](../code/..)" → 去掉链接留纯文字
        line = re.sub(r"\[`?code/series-" + re.escape(series) + r"\.py`?\]\(\.\./code/series-"
                      + re.escape(series) + r"\.py\)", "本文文末《完整可跑代码》", line)
        # 兜底：仍残留的裸路径
        line = line.replace(f"../code/series-{series}.py", "文末《完整可跑代码》")
        line = line.replace(f"code/series-{series}.py", f"series-{series}.py")
        return line
    text = dead_link_re(series).sub(repl, text)

    # 3) 在 "## 5" 之前插入完整代码块
    block = build_block(series, code)
    m = re.search(r"^---\s*\n\s*## 5\.", text, flags=re.MULTILINE)
    if m:
        insert_at = m.start()
        text = text[:insert_at] + block + "\n\n" + text[insert_at:]
    else:
        # 没找到就追加到文末
        text = text.rstrip() + "\n\n---\n\n" + block + "\n"

    md_path.write_text(text, encoding="utf-8")
    return f"✅ {series}: 已内嵌 {len(code.splitlines())} 行代码"


def main():
    targets = sys.argv[1:] or SERIES
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for s in targets:
        print(process(s))


if __name__ == "__main__":
    main()
