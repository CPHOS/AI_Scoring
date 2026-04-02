# CPHOS AI 自动阅卷系统

基于视觉语言模型（VLM）和大语言模型（LLM）的物理竞赛手写答卷自动评分系统。系统将 **视觉感知**（手写识别）与 **逻辑推理**（评分对齐）解耦，以提高准确度并降低 token 成本。

> **当前阶段：** Phase 1 — 感知层（VLM 转录）与配套工具链已完成初版。

---

## 项目结构

```
CPHOS/
├── VLM-converter/          # VLM 感知层：手写图片 → LaTeX Markdown
│   ├── main.py             # CLI 入口
│   ├── requirements.txt
│   └── vlm_converter/
│       ├── config.py           # 环境变量 / 默认配置
│       ├── input_processing.py # 图片加载与 base64 编码
│       ├── latex_normalizer.py # LaTeX 文本规范化
│       ├── openrouter_client.py# OpenRouter API 调用
│       ├── prompt_builder.py   # System / User prompt 构建
│       ├── request_id.py       # 请求 ID 生成
│       ├── response_parser.py  # API 响应文本提取
│       ├── service.py          # 编排层：单次/批量转录 → Markdown 输出
│       ├── types.py            # 数据类型定义
│       └── validation.py       # 转录结果校验
├── std-sol-converter/      # 标准答案解析器：CPHOS LaTeX 评分标准 → JSON
│   ├── cphos_latex_to_json.py
│   └── parsed_output.json
├── student-viewer/         # 转录可视化：JSON → PDF（原图 + 渲染公式对照）
│   └── render.py
├── overall-plan.md         # 总体架构设计文档
└── .env.example            # 环境变量模板
```

---

## 快速开始

### 环境要求

- Python 3.11+
- [tectonic](https://tectonic-typesetting.github.io/)（仅 student-viewer PDF 编译需要）

### 安装

```bash
cd CPHOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r VLM-converter/requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 OpenRouter API Key
```

### VLM 转录

将手写答卷图片转录为 LaTeX Markdown 文档。每张图片生成一个 `.md` 文件，YAML frontmatter 中包含来源图片、模型信息和 token 用量。

```bash
# 单张图片
python3 VLM-converter/main.py image.png -o result.md

# 批量处理（每张图片 → 一个 .md，外加 _summary.json）
python3 VLM-converter/main.py --input-dir ./test-figs/student -o ./output/
```

**输出格式（`.md` 文件）：**

```markdown
---
source_files: ["微信图片_xxx.png"]
request_id: "20260402192017-55-19"
model: "openai/gpt-4o-mini"
provider: "openrouter"
timestamp: "2026-04-02T12:00:00Z"
prompt_tokens: 1234
completion_tokens: 567
total_tokens: 1801
generation_id: "gen-xxx"
---

第 28 届 CPHOS 物理专题考（波动与光学）

题目 (1) $$M_{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$
...

## 被忽略内容
- 非相关脚注（划线内容，略）
```

### 标准答案解析

将 CPHOS LaTeX 评分标准文件解析为结构化 JSON，供下游评分 Agent 使用。

```bash
python3 std-sol-converter/cphos_latex_to_json.py input.tex -o parsed.json
```

### 转录可视化（PDF）

将旧版 JSON 转录结果渲染为 PDF，每位学生一页：上方原始手写图、下方 LaTeX 渲染公式。

```bash
# 使用 tectonic 编译
python3 student-viewer/render.py students_trans.json -o output.pdf

# 仅生成 .tex 文件（调试用）
python3 student-viewer/render.py students_trans.json --tex-only --work-dir ./build
```

---

## 系统架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   手写图片    │────▶│  VLM 感知层   │────▶│  Markdown    │
│  (PNG/JPEG)  │     │ (GPT-4o-mini)│     │  (LaTeX 转录) │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐             ▼
│  LaTeX 评分   │────▶│ 标答解析器    │────▶┌──────────────┐
│  标准 (.tex)  │     │              │     │  LLM 评分层   │
└──────────────┘     └──────────────┘     │  (Phase 2)   │
                                          └──────────────┘
```

| 层级 | 组件 | 功能 | 模型 |
|:---|:---|:---|:---|
| **感知** | VLM-converter | 手写图片 → LaTeX 文本 | GPT-4o-mini |
| **推理** | Grading Agent（规划中） | 文本对齐评分标准 → 打分 | GPT-4o / Claude |
| **输出** | Report Generator（规划中） | 生成教师可读评分报告 | Python |

---

## 配置

通过 `.env` 文件或环境变量配置：

| 变量 | 说明 | 默认值 |
|:---|:---|:---|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | （必填） |
| `OPENROUTER_MODEL` | 使用的模型 | `openai/gpt-4o-mini` |
| `OPENROUTER_BASE_URL` | API 端点 | `https://openrouter.ai/api/v1` |
| `OPENROUTER_TIMEOUT_SECONDS` | 请求超时 | `120` |
| `OPENROUTER_MAX_RETRIES` | 最大重试次数 | `2` |

---

## 许可证

MIT License © 2026 gameswu & CPHOS