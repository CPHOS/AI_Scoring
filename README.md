# CPHOS AI 自动阅卷系统

基于视觉语言模型（VLM）和大语言模型（LLM）的物理竞赛手写答卷自动评分系统。系统将 **视觉感知**（手写识别）与 **逻辑推理**（评分对齐）解耦，以提高准确度并降低 token 成本。

---

## 项目结构

```
AI_Scoring/
├── pyproject.toml              # 项目配置（uv 管理依赖）
├── .env.example                # 环境变量模板
├── README.md
└── src/
    ├── __init__.py
    ├── __main__.py             # CLI 入口
    ├── config.py               # 环境变量 / Settings 管理
    ├── client/                 # 大模型服务商客户端
    │   ├── __init__.py
    │   └── openrouter.py       # OpenRouter API 客户端
    ├── model/                  # 公用数据类型
    │   ├── __init__.py
    │   └── types.py            # InputAsset, TranscriptionResult
    ├── judge/                  # 判卷模块（待实现）
    │   └── __init__.py
    └── recognize/              # 卷面识别：手写图片 → LaTeX Markdown
        ├── __init__.py
        ├── input_processing.py # 图片/PDF 加载与 base64 编码
        ├── latex_normalizer.py # LaTeX 文本规范化
        ├── prompt_builder.py   # System / User prompt 构建
        ├── request_id.py       # 请求 ID 生成
        ├── response_parser.py  # API 响应文本提取
        ├── service.py          # 编排层：单次/批量转录
        └── validation.py       # 转录结果校验
```

---

## 快速开始

### 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（推荐的包管理器）

### 安装

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并进入项目
cd AI_Scoring

# 同步依赖（自动创建虚拟环境）
uv sync

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 OpenRouter API Key
```

### 卷面识别（VLM 转录）

将手写答卷图片转录为 LaTeX Markdown 文档。

```bash
# 单张图片
uv run python -m src image.png -o result.md

# 批量处理（每张图片 → 一个 .md，外加 _summary.json）
uv run python -m src --input-dir ./test-figs/student -o ./output/
```

**输出格式（`.md` 文件）：**

```markdown
---
source_files: ["微信图片_xxx.png"]
request_id: "example-id"
model: "google/gemini-2.5-flash-lite"
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

---

## 系统架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   手写图片    │────▶│  recognize    │────▶│  Markdown    │
│  (PNG/JPEG)  │     │ (VLM 卷面识别)│     │  (LaTeX 转录) │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │   judge       │
                                          │ (LLM 判卷)    │
                                          │  (规划中)      │
                                          └──────────────┘
```

| 模块 | 职责 | 状态 |
|:---|:---|:---|
| `client/` | 管理与大模型服务商的连接 | ✅ 已完成 |
| `model/` | 公用数据类型定义 | ✅ 已完成 |
| `recognize/` | 手写图片 → LaTeX 转录 | ✅ 已完成 |
| `judge/` | 文本对齐评分标准 → 打分 | 🚧 待实现 |

---

## 配置

通过 `.env` 文件或环境变量配置：

| 变量 | 说明 | 默认值 |
|:---|:---|:---|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | （必填） |
| `OPENROUTER_MODEL` | 使用的模型 | `google/gemini-2.5-flash-lite` |
| `OPENROUTER_BASE_URL` | API 端点 | `https://openrouter.ai/api/v1` |
| `OPENROUTER_TIMEOUT_SECONDS` | 请求超时（秒） | `120` |
| `OPENROUTER_MAX_RETRIES` | 最大重试次数 | `2` |

---

## 许可证

MIT License © 2026 gameswu & CPHOS