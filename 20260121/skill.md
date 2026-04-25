## 核心设计理念:Progressive Disclosure(渐进式信息披露)
> 底层动机:Context Window 是个"稀缺资源"

### 传统方式:
```
┌─────────────────────────────────────┐
│ System Prompt (5K tokens)          │
│ + RAG 检索结果 (10K tokens)        │
│ + 所有 Tool 定义 (8K tokens)       │
│ + 用户问题 (1K tokens)             │
│ = 24K tokens 已经用了!             │
└─────────────────────────────────────┘
Claude: "我还没干活呢就快撑死了..."
```

### Agent Skills: 
```
启动时:
┌─────────────────────────────────────┐
│ System Prompt (2K tokens)          │
│ + Skills 元数据 (500 tokens)       │  ← 只加载"目录"
│   - pdf-skill: 处理PDF文档          │
│   - excel-skill: 分析电子表格       │
│   - brand-skill: 品牌设计指南       │
│ + 用户问题 (1K tokens)             │
│ = 3.5K tokens                      │
└─────────────────────────────────────┘

用户:"帮我从 report.pdf 提取表格"
Claude: "哦?需要用 pdf-skill!"
        [动态读取 /skills/pdf/SKILL.md]
        [只加载 PDF 相关指令 2K tokens]
        
总消耗: 5.5K tokens (省了 70%!)
```

### Progressive Disclosure:
```
第一层:轻量级元数据(name, description)→ 帮 Agent 决策"要不要用这个 Skill"
第二层:完整指令(SKILL.md 主体)→ 只在触发时加载
第三层:附加资源(scripts, templates)→ 按需执行
```

### **vs. Function Calling / MCP (Model Context Protocol)**

| 特性 | Function Calling | Agent Skills |
| --- | --- | --- |
| **粒度** | 细粒度操作(原子工具) | 粗粒度流程(任务模板) |
| **上下文** | 所有工具定义提前加载 | 按需加载指令 |
| **适用** | 调用外部服务(API) | 传授内部知识(流程) |
| **举例** | `send_email()` | "如何写周报" |

**直觉类比**:

- **MCP/Tools**: 给员工一把"瑞士军刀"(锤子、螺丝刀、刀片...)
- **Skills**: 给员工一本"IKEA 组装手册"(先拧这个,再装那个...)

**它们是互补的**:

```
Skills 里可以调用 MCP Tools!
┌─────────────────────────────┐
│ pdf-skill/SKILL.md          │
│ "用 python-tool 执行:       │
│  import PyPDF2              │
│  pdf = PyPDF2.PdfReader..." │
└─────────────────────────────┘


MCP = 硬件接口(连接外部世界)
Skills = 软件包(封装内部知识)

┌─────────────────────────────────┐
│         Claude Agent            │
│  ┌──────────┐   ┌──────────┐   │
│  │ MCP 连接  │   │ Skills   │   │
│  │ 外部工具  │   │ 内部知识  │   │
│  └──────────┘   └──────────┘   │
└─────────────────────────────────┘

```


```
# 系统启动时:
system_prompt = """
你是个助手。
可用 Skills:
- pdf_skill: 处理 PDF 文件
- excel_skill: 处理 Excel 文件  
- ppt_skill: 处理 PPT 文件
"""  # 只占用 200 tokens!

# 用户:"帮我填 PDF 表单"
# → Claude 读取 pdf_skill/SKILL.md (2000 tokens)
# → 如果需要表单填写,再读取 pdf_skill/forms.md (1000 tokens)
```

**对比表**:

| 方式 | 启动 tokens | 任务 tokens | 总计 |
|------|------------|------------|------|
| 传统 | 50k | 0 | 50k |
| Skills | 0.2k | 3k(仅相关) | 3.2k |
| **节省** | **99.6%** | - | **93.6%** |

---

#### 突破 2: **Files as Context**(文件即上下文)

**核心思想**: 把 AI 需要的知识存成文件,而不是塞进 Prompt

**类比**(绝妙!):
```
传统 Agent = 背诵整本操作手册的员工(容易累,容易忘)
Agent Skills = 有操作手册可查的员工(高效,可扩展)
```

```
my-skill/
├── SKILL.md          # 主文档(Always loaded)
├── reference.md      # 参考资料(On-demand)
├── forms.md          # 表单填写指南(On-demand)
└── scripts/
    └── extract.py    # 可执行脚本
 ```   

<img width="1650" height="929" alt="image" src="https://github.com/user-attachments/assets/f1f05c5c-512f-4d6d-b789-73a7bd47edd2" />

### SKILL.md
name 和 description


<img width="1650" height="1069" alt="image" src="https://github.com/user-attachments/assets/3df7ac44-0e41-48d7-af00-e384eacc6559" />

#### two additional files (reference.md and forms.md)





#### scripts
<img width="1650" height="929" alt="image" src="https://github.com/user-attachments/assets/18e07679-dc59-417d-b5b1-85e5c31891ae" />

MCP = Agent 的"手"(调用外部工具)
Skills = Agent 的"大脑"(知道如何使用这些工具)

例子:
- MCP: 连接 Jira API,获取 issue 列表
- Skill: "如何根据 issue 生成周报"(使用 Jira 数据的流程)

## 场景 
### **个人工作流**

```bash
my-skills/
├── blog-writing/
│   ├── SKILL.md  # SEO 优化指南
│   └── templates/文章模板.md
├── meeting-notes/
│   ├── SKILL.md  # 会议记录格式
│   └── templates/周会模板.md
└── code-review/
    ├── SKILL.md  # 我的代码审查标准
    └── checklist.md
```

---

### 参考文档：
1. [anthropic](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
