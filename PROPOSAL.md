# 基于 ESMC-SAE 稀疏特征库的蛋白质概念搜索引擎

> **开题报告 v4.3 · 2026-06-08** (覆盖 v4 / v4.1 / v4.2 / v3)
> 课题方向:从"序列/结构比对"到"稀疏特征布尔检索"的范式转移
> 论文主轴:方法论创新(投稿目标:Bioinformatics / Nature Methods)
> 案例方向:塑料降解酶(PETase / MHETase)远缘同源体挖掘
> 核心约束:**零 GPU,纯数据工程,云端截胡,本地极速检索**

---

## 0. 摘要

本课题利用 EvolutionaryScale 2026 年公开的 ESMC 6B + 16,384 维稀疏自编码器(SAE),把每个蛋白质映射为"概念组合",并基于 **Elasticsearch 倒排索引** 构建**支持布尔查询的在线搜索引擎**。与论文(Candido et al. 2026)已发表的"点积相似度 + Foldseek 离线验证"范式相比,本工作的核心差异是把"特征库"从离线分析工具升级为**可在线做概念组合查询的搜索引擎**,例如:`(hydrolase-active-site) AND (alpha/beta-hydrolase-fold) AND NOT (signal-peptide)`。

**关键工程取舍**:首期以 57 万 SwissProt 蛋白为索引对象(经 AWS Athena JOIN 拉取,50MB 数据),零 GPU、几美元成本、单台笔记本 + Docker 即可跑通。索引 schema 设计上保留对 6.8B 全量 Atlas 的扩展能力,但首期不做暗物质挖掘(后续工作扩展到 MGnify/SPIRE 元基因组库)。

---

## 1. 课题背景与意义

### 1.1 蛋白质功能检索的现状与痛点

传统蛋白质功能预测有三条主流路径,各有不可弥补的局限:
- **序列比对**(BLAST、MMseqs2):远缘同源(< 30% identity)时召回率断崖式下跌;
- **结构比对**(Foldseek、TM-align):对无结构预测的蛋白不可用,对单点突变敏感;
- **深度学习表征 + 向量 ANN**(ESM-2 + FAISS):本质是"相似度检索",**无法做概念组合查询**——用户难以表述"我要找同时具备 α/β-水解酶折叠 + 活性位点丝氨酸 + 无信号肽的蛋白"。

第三条路径目前缺少的是**布尔组合查询接口**。这导致两类痛点长期未解决:
- 药物脱靶、酶工程改造等场景需要"在数十亿蛋白中找满足一组生物学约束的子集",传统工具无法直接表达这种约束;
- 工业酶筛选时,序列相似度搜索因同源性低而失效,暗物质蛋白的发现只能靠随机宏基因组测序。

### 1.2 ESMC-SAE 范式转移的契机

2026 年 1 月,Candido 等人发表 ESMC 模型与配套 SAE,把蛋白质 LLM 的 2560 维隐藏层激活通过 384 维 bottleneck 的 TopK(K=64)稀疏自编码器,**显式分解为 16,384 个可独立解释的"基础生物学概念"**。论文披露的关键事实:

- 每个残基激活 64 个 feature(layer 60 of ESMC 6B);
- 195,000 个 SwissProt 参考蛋白经多 agent GPT-5 标注,每个 feature 获得生物学描述,覆盖 8 大类(残基身份、二级结构、三级模体、域/折叠、无序/低复杂度、生化微环境、定位/拓扑、功能位点);
- **6.8B 蛋白全量 SAE 激活已被计算并发布在 AWS Open Data(Athena 可查)**;
- Jaccard ≥ 0.6 聚类产生 230M 簇(≥5 成员)+ 7.7M 簇(≥50 成员);
- SAE 特征相似度在远缘同源(< 40% identity)情形下显著优于序列/结构比对(图 S39 / S40)。

**核心叙事**:SAE 把"特征库"变成了可独立使用的第一公民。但论文的检索范式是"点积相似度 + Foldseek 验证"——**没有提供在线可查询的布尔接口**。这正是本课题要补上的那一块拼图。

### 1.3 工程哲学:Cloud-Native Zero-GPU

本课题**刻意回避"用 GPU 集群硬算"的常规路线**,采用"**云端截胡 + 本地极速检索**"的零 GPU 架构:
- **云端**:AWS Athena 直接对 6.8B 蛋白的特征表做 SQL JOIN,提取 SwissProt 子集(57 万条);
- **下载**:50MB CSV/JSONL,几美元成本;
- **本地**:单台笔记本 + Docker 跑 Elasticsearch,Roaring Bitmap 算法 0.01s 响应布尔查询;
- **算力门槛**:**零 GPU,任何课题组都能复现**。

这条路线回应"计算平权":大厂的算力(ESMC 6B 推理 + 6.8B 激活计算)已经替我们做完,我们要做的是"**让这些数据能被人用起来**"。

---

## 2. 国内外研究现状

### 2.1 蛋白质 LLM 与稀疏特征解耦

| 里程碑 | 团队 / 时间 | 关键贡献 |
|---|---|---|
| ESM-2 | Meta FAIR 2022 | 650M-15B 蛋白 LLM,产生稠密 embedding |
| ProGen | Salesforce 2023 | 序列生成式 LLM |
| **ESMC 6B + SAE** | **EvolutionaryScale 2026** | **蛋白 LLM + 16,384 稀疏概念,layer 60 解耦** |
| InterPLM | Kemp et al. 2024 | ESM-2 上独立开发的 SAE,平行工作 |
| MechRepo | Bose et al. 2025 | 尝试用 LLM 表征做机制级功能预测 |

**现状判断**:SAE-as-features 范式已建立,但"特征—蛋白"在线检索的工程实现目前**只有论文作者在内部用**(A.5.4 描述的 MinHash + Spark 离线聚类,无在线 query 入口)。

### 2.2 蛋白质相似度检索工具

| 工具 | 范式 | 局限 |
|---|---|---|
| BLAST / MMseqs2 | 序列 k-mer | 远缘同源失败 |
| Foldseek / TM-align | 结构比对 | 无结构预测时不可用 |
| ESM-2 / ProGen + FAISS | 稠密向量 ANN | 无布尔组合 |
| ProtENN | embedding + 编辑距离 | 同上 |
| **本课题** | **稀疏特征 + 倒排索引 + 布尔** | **首创可在线查询的概念组合** |

### 2.3 倒排索引在生物信息学的应用

倒排索引在文献检索(PubMed)与基因区间查询(IGV)中成熟,但在**蛋白质功能检索**领域几乎空白。本课题是把"term → document"映射平移到"feature → protein_id"。

---

## 3. 研究目标与内容

### 3.1 总目标

构建一个**面向生物学家日常使用**的蛋白质概念搜索引擎,核心能力是:用户用 8 大类特征(残基、二级结构、模体、域、无序、微环境、定位、功能位点)自由组合,**亚 10 毫秒级**返回满足约束的蛋白集合。

### 3.2 具体目标

1. **数据层**:Athena JOIN 拉取 SwissProt 57 万条 + 8 类别 feature 字典;
2. **索引层**:Elasticsearch `keyword` 数组 + Roaring Bitmap 倒排索引;
3. **查询层**:RESTful API + Web 检索界面,支持 `must` / `should` / `must_not` 布尔查询;
4. **评估层**:8 大特征类别下 precision@10 / recall@100 评测,对照 SAE dot product、MMseqs2、Foldseek;
5. **案例层**:塑料降解酶家族远缘同源挖掘。

### 3.3 与论文已发表工作的边界

| 维度 | 论文(Candido et al. 2026) | 本课题 |
|---|---|---|
| SAE 激活 | 全量计算(6.8B 蛋白) | **不重算,直接 Athena 截胡** |
| 聚类 | MinHash + Spark,生成 230M 簇 | **不重复**,首期不需要 |
| 检索 | 点积相似度 + Foldseek 验证 | **布尔概念查询 + 倒排索引** |
| 用户接口 | 无(离线分析) | **在线可查询引擎** |
| 评估 | GO BP / biome 富集 | **precision@10 / recall@100,8 大类别** |
| 案例 | RNase H / Cas12 / TnpB | **塑料降解酶家族** |
| 算力 | H100 集群 | **零 GPU** |

---

## 4. 研究方法与技术路线

### 4.1 总体技术架构

```
┌─────────────────────────────────────────────────────────────┐
│       AWS Open Data: ESM Atlas (S3 + Athena)                │
│  esm_public_atlas 表: protein_id → sae_features[]           │
│  (论文作者已用 ESMC 6B + SAE 计算了 6.8B 蛋白的激活)         │
└───────────────────────────┬─────────────────────────────────┘
                            │  Athena JOIN (~2 min, < $5)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  P1: 零 GPU 数据获取(15 分钟)                               │
│  - UniProt 下载 SwissProt Entry ID 列表(57 万条)            │
│  - 上传 S3 → Athena 建表 → JOIN esm_public_atlas            │
│  - 下载 ~50MB CSV/JSONL 到本地                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  P2: 本地 ES 倒排索引(30 分钟)                              │
│  - Docker 启动单节点 ES 8.10                                 │
│  - 定义 mapping: features → keyword 数组                    │
│  - Python helpers.bulk 灌库(50 万条,几秒)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  P3: 查询引擎 + Web 检索界面                                 │
│  - REST API: POST /search {must, should, must_not}          │
│  - 前端: feature 树形选择 + 拖拽组合                          │
│  - 评估: 8 大类别 precision@10 / recall@100                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  P4: 塑料降解酶案例                                          │
│  - 已知 PETase → 提取 SAE feature 组合                       │
│  - Boolean query 命中 57 万 SwissProt                        │
│  - α/β-水解酶超家族远缘同源体挖掘                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  P5: 论文与开源                                              │
│  - GitHub + Zenodo 全栈开源                                 │
│  - 投稿 Bioinformatics / Nature Methods                      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 P1 · Athena 截胡数据(2 周)

**输入**:UniProt SwissProt `reviewed:true` 全量(57 万条)

**关键操作**:

**Step 1: 准备猎物清单**(10 分钟)
- UniProt 官网搜索 `reviewed:true`,TSV 格式下载,只保留 `Entry` 列
- ~570K 个 UniProt ID(如 P12345)
- 上传 S3:`s3://my-protein-bucket/target_ids.csv`

**Step 2: Athena 建表映射**(2 分钟)
```sql
CREATE EXTERNAL TABLE IF NOT EXISTS my_swissprot_ids (
  uniprot_id STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
LOCATION 's3://my-protein-bucket/';
```

**Step 3: 神之 JOIN**(2-5 分钟)
```sql
SELECT 
    atlas.protein_id, 
    atlas.sae_features,   -- 论文已算好的稀疏特征数组
    atlas.activations     -- (可选)每个 feature 的激活强度
FROM 
    esm_public_atlas AS atlas
INNER JOIN 
    my_swissprot_ids AS target 
ON 
    atlas.protein_id = target.uniprot_id;
```

**Step 4: 下载战利品**(5 分钟)
- Athena 结果存到 S3,下载 ~50MB CSV/JSONL
- 包含:570K 条 (protein_id, sae_features[]) 记录

**Step 5: 数据验证**(2-3 天)
- 与论文 A.4.5.3 报告的"激活分布"对一对
- 抽 1000 条做 sanity check(高激活 feature 是否对应已知 Pfam 域?)
- **技术报告 1**:数据获取流水与字段定义

**P1 的不确定性**:ESM Atlas 公开的 `esm_public_atlas` 表的字段细节(sae_features 的具体格式、是否包含权重、是否覆盖所有 SwissProt)需要查 EvolutionaryScale 官方文档;若字段命名或 schema 与上述假设不同,Step 3 的 SQL 略作调整。

### 4.3 P2 · ES 倒排索引(2-3 周)

**Step 1: Docker 启动 ES**(1 分钟)
```bash
docker run -d --name protein-search -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.10.2
```

**Step 2: 索引 mapping**(1 分钟)
```bash
curl -X PUT "localhost:9200/sae_proteins" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "protein_id": { "type": "keyword" },
      "length":     { "type": "integer" },
      "features":   { "type": "keyword" }    // 核心:keyword 数组自动生成倒排索引
      // "feature_weights": { "type": "float" }  // 可选,用于加权评分
    }
  }
}'
```

**Step 3: 数据灌库**(1-5 分钟)
```python
import json
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200")

def generate_data():
    with open("athena_results.jsonl", "r") as f:
        for line in f:
            doc = json.loads(line)
            yield {
                "_index": "sae_proteins",
                "_id": doc["protein_id"],
                "_source": {
                    "protein_id": doc["protein_id"],
                    "length": doc.get("length", 0),
                    "features": doc["sae_features"]  # e.g. ["102", "8092", "405"]
                }
            }

helpers.bulk(es, generate_data())
```

**Step 4: 性能基线**(2-3 天)
- 50 万 / 100 万 / 500 万 / 6.8B 扩展性测试
- 不同 query 复杂度(1 个 must / 3 个 must / 5 must + 3 must_not)
- 测量 P99 延迟、ES 堆内存、磁盘占用

**Step 5: 索引优化**(可选,2-3 天)
- IPF 过滤(去掉在 > 80% 蛋白出现的"非判别性"feature,论文 A.4.2.4 阈值 87%)
- BM25 评分调参
- Posting list 压缩参数调优

**产出**:
- `index/es_schema.json`、 `index/build_index.py`
- **技术报告 2**:PoC 性能基线

### 4.4 P3 · 查询引擎 + 评估(6-8 周)

**4.4.1 REST API**
```python
# POST /search
{
  "must": [102, 8092, 405],        # 全部必须具备
  "should": [1024, 2048],          # 至少一个具备(加分)
  "must_not": [9999, 1234],        # 全部必须不具备
  "size": 20,
  "explain": false
}
```

**4.4.2 Web 检索界面**(可选)
- 8 大特征类别树形结构
- 用户拖拽 feature 到 must / should / must_not 区域
- 实时显示命中数(随用户编辑即时更新)
- 结果列表:protein_id、长度、命中 feature、点击跳转 UniProt

**4.4.3 评估基准**
- **已知关系**:论文 S44 的 70 个 Pfam 类别(Ser/His/Asp proteases、ABC transporter、K+ channel 等)
- **功能层级**:8 大类别各取 10-20 个 feature 组合查询
- **对照方法**:
  - A. SAE Jaccard(论文 A.4.5 范式,我们的"离线版")
  - B. MMseqs2(序列相似度)
  - C. Foldseek(结构相似度)
  - D. ESM-2 cosine(稠密向量 ANN)

**4.4.4 评估指标**
- `precision@10`:前 10 个结果中已知相关的比例
- `recall@100`:前 100 个结果覆盖已知相关的比例
- **布尔查询特有**:`constraint_satisfaction_rate` = 严格满足 must AND NOT must_not 的比例
- 配对 Wilcoxon 符号秩检验,p < 0.01

**产出**:
- `eval/benchmark.py`、`eval/results_8categories.md`
- **技术报告 3**:方法论评测(投稿核心)

### 4.5 P4 · 塑料降解酶案例(8-12 周)

**4.5.1 生物学背景**
- PETase(Ideonella sakaiensis, 2016):已知高效降解 PET 塑料
- MHETase:PETase 协同酶,水解中间产物 MHET
- α/β-水解酶超家族:核心折叠,催化三联体 Ser-His-Asp
- **科学问题**:在 57 万 SwissProt 中,是否存在已注释为"非 PETase"但 SAE 特征上"很像 PETase"的蛋白?这些可能是被遗漏的远缘同源体。

**4.5.2 查询构造**
- 已知 PETase / MHETase 序列(从 SwissProt + 文献,~500-1000 个)
- 算 SAE 激活(ESMC 6B + SAE,**这步需要 GPU,但量小,半天可完成**)
- 取高激活 feature,形成 query 模板
- **关键 feature 类别**:
  - α/β-水解酶折叠相关 ~50 个
  - 催化三联体微环境 ~10 个
  - 底物结合口袋 ~10 个
  - 必须 NOT 的 ~5 个(信号肽、跨膜螺旋等)

**4.5.3 Boolean query 模板**
```
must:
  - α/β-hydrolase fold features (~50)
  - catalytic triad microenvironment (~10)
  - substrate-binding pocket (~10)
should:
  - PETase-specific motif (~5)
must_not:
  - signal peptide features (~5)
  - transmembrane helix features (~3)
```

**4.5.4 候选评估**
- 命中 ~500-5000 个 SwissProt 蛋白
- 二次过滤:
  - 序列长度合理(200-500 aa)
  - Pfam 域为 α/β-水解酶家族(PF00561 / PF12697)
  - ESMFold2 结构已有
- 候选数:**~50-200 个高置信候选**
- 文献交叉验证:是否有任何文献已提及"该蛋白具有 PET 水解活性"?

**4.5.5 暗物质版(可选,推后到 P5 之后)**
- **未来工作**:把数据源从 SwissProt 扩展到 MGnify + SPIRE + JGI(元基因组暗物质)
- 同一查询在 6.8B 全量上跑,挖"无 Pfam 注释但 SAE 特征像 PETase"的候选
- 需要重新跑一次 Athena JOIN,查询时间不变

**产出**:
- `case/petase_query.json`、`case/petase_candidates.fasta`
- **技术报告 4**:案例研究完整报告

### 4.6 P5 · 论文与开源(8-12 周)

- GitHub 仓库:`esmc-search`(MIT 协议)
- 数据流脚本 + 索引构建 + 查询 API + Web UI
- Zenodo DOI
- 预印本:bioRxiv
- 投稿:**Bioinformatics**(方法论主轴)/ 备选 Nature Methods

---

## 5. 创新点

1. **首创"稀疏特征 + 倒排索引"蛋白质检索范式**。把 LLM 表征从"相似度工具"升级为"概念组合查询系统",无先例。
2. **Cloud-Native Zero-GPU 架构**。Athena 截胡 + ES 倒排,零 GPU 几美元跑通,完全摆脱"必须有大集群才能做大模型研究"的限制。
3. **8 大类别统一评估框架**。把"特征库"与"GO / Pfam"等经典注释对齐,提供可比较的 precision@10 / recall@100。
4. **塑料降解酶家族的系统性远缘挖掘**。在已知 57 万 SwissProt 中找潜在 PETase,工业价值清晰。

---

## 6. 进度安排

| 阶段 | 时间 | 关键交付 | 算力需求 |
|---|---|---|---|
| **P1** 数据获取 | 第 1-2 周 | Athena JOIN 流水、50MB 数据集、技术报告 1 | 笔记本 + AWS 账户 |
| **P2** 倒排索引 PoC | 第 3-5 周 | 索引构建脚本、查询 API、性能基线 | 笔记本 + Docker |
| **P3** 评估 | 第 6-13 周 | 8 大类别评测、Web UI、技术报告 3 | 笔记本 + Docker |
| **P4** 案例研究 | 第 14-25 周 | 候选蛋白、活性分析、技术报告 4 | 笔记本 + Docker + 半天 GPU(算 PETase 激活) |
| **P5** 论文开源 | 第 26-37 周 | 预印本、投稿、release | 无新增 |

**总预算(570K 版)**:**< 1000 RMB** — 主要是 AWS Athena 查询费($0.05) + 半天的 GPU 租用(P4 算 PETase 激活,~1000 RMB)。

> **注**:这是 v4 的 570K SwissProt 笔记本版预算。若走 v4.2 的 6.8B 全量版,见 §11.7,预算为 **3-4 万一次性(自建 8TB SSD)/ 1.5 万短跑(阿里云 1 台)**。

---

## 7. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|---|---|---|---|
| `esm_public_atlas` 表字段与假设不符 | 中 | 高 | Step 3 SQL 略调;查 EvolutionaryScale 官方文档确认 schema |
| Athena JOIN 不支持数组字段 | 低 | 高 | 改用 S3 Select + 本地 Python merge |
| 57 万规模下查询结果太少(< 10) | 中 | 中 | 放松 must_not;加入 more_like_this;查阅 feature 字典确认 query 合理 |
| 57 万规模下查询结果太多(> 10 万) | 低 | 低 | 加更多 must 约束;P4 阶段做 |
| 塑料降解酶候选全部已知 | 中 | 中 | 老实报告"SwissProt 内的远缘挖掘边界",把"暗物质版"作为 future work 强调 |
| 投稿被拒 | 中 | 中 | 主投 Bioinformatics,备选 Nature Methods / PLOS Comp Biol |

---

## 8. 与 v3 方案的关键差异(版本说明)

| 维度 | v3 (已弃) | v4 (当前) |
|---|---|---|
| 首期数据规模 | 7.7M cluster representative | **57 万 SwissProt** |
| 算力 | 8×A100 × 10 天 | **零本地 GPU**(Athena 截胡 + 拉现成数据) |
| 成本 | 35-80 万 RMB | **3-4 万一次性(自建 8TB SSD)/ 1.5 万短跑(阿里云 1 台)** |
| 数据获取 | 拉全量 + 自算 representative | **Athena JOIN + 50MB 下载** |
| 暗物质叙事 | 首期承诺 | **推到 future work** |
| 起跑时间 | 1-2 个月 | **当周可出结果** |

---

## 9. 参考文献(初稿)

1. **Candido, M. J., et al. (2026).** *Language Modeling Materializes a World Model of Protein Biology.* EvolutionaryScale / Biohub. (`/Users/huyue/esmc_search/esm_protein.pdf`)
2. **Lin, Z., et al. (2023).** *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science, 379(6637), 1123-1130.
3. **van Kempen, M., et al. (2024).** *Fast and accurate protein structure search with Foldseek.* Nature Biotechnology, 42, 243-246.
4. **Steinegger, M., & Söding, J. (2017).** *MMseqs2: sensitive protein sequence searching for the analysis of massive data sets.* Nature Biotechnology, 35, 1026-1028.
5. **Yoshida, S., et al. (2016).** *A bacterium that degrades and assimilates poly(ethylene terephthalate).* Science, 351(6278), 1154-1156.
6. **Palm, G. J., et al. (2019).** *Structure of the plastic-degrading Ideonella sakaiensis MHETase bound to a substrate.* Nature Communications, 10, 1717.
7. **Bileschi, M. L., et al. (2022).** *Using deep learning to annotate the protein universe.* Nature Biotechnology, 40, 932-937.
8. **Cunningham, F., et al. (2022).** *Ensembl 2022.* Nucleic Acids Research, 50(D1), D988-D995.
9. **Richardson, L., et al. (2023).** *MGnify: the microbiome sequence data analysis resource in 2023.* Nucleic Acids Research, 51(D1), D753-D759.
10. **Chen, I.-M. A., et al. (2023).** *The IMG/M data management and analysis system v.7.* Nucleic Acids Research, 51(D1), D739-D748.
11. **Schmidt, T. S. B., et al. (2024).** *SPIRE: a searchable, planetary-scale, rapid investigation of the metagenomic universe.* (EMBL SPIRE)
12. **Rath, S., et al. (2021).** *MitoCarta3.0: an updated mitochondrial proteome — now with full-body organ expression.* Nucleic Acids Research, 49(D1), D1541-D1547.
13. **Kanehisa, M., et al. (2023).** *KEGG for taxonomy-based analysis of pathways and genomes.* Nucleic Acids Research, 51(D1), D587-D592.
14. **Jumper, J., et al. (2021).** *Highly accurate protein structure prediction with AlphaFold.* Nature, 596, 583-589.

---

## 10. 附录:事实锚定(对应论文 A.4-A.6)

| 方案中的事实 | 论文出处 |
|---|---|
| ESMC 6B + SAE 公开,可在 Athena 拉取 | A.6 |
| 16,384 codebook, K=64, layer 60 | A.4.1, S11 |
| 6.8B 蛋白全量 SAE 激活 | A.5.1 |
| 8 大特征类别 | A.4.3.1, S12 |
| 多 agent GPT-5 标注 195K SwissProt | A.4.3.1 |
| Jaccard ≥ 0.6 聚类 → 230M / 7.7M 簇 | A.5.4.5 |
| IPF 过滤(87% 阈值去掉 3,635 feature) | A.4.2.4 |
| 70 Pfam 类别评估基准 | S44 |
| 8 个数据源(UniRef / JGI / MGnify / SPIRE) | S13 |
| 在 < 40% 序列一致度下 SAE 特征优势 | S39, S40 |

---

**下一步动作**:
1. PI 审阅 v4.3,确认路径 Y(7.7M representative)是否接受;
2. 若接受,**当周启动 P1a**:在 AWS 账户跑一次 Athena JOIN,验证 `esm_public_atlas` 的字段结构。

---

## 11. 6.8B 全量扩展可行性分析(v4.3 增量)

> **触发**:PI 提供 EvolutionaryScale 官方 HuggingFace / Biohub 页面信息,披露 **ESM Atlas 数据集的真实规模与下载方式**。**v4.3 根本性修正**:
>
> 1. **数据不在 HuggingFace,在 S3 公开桶**:`s3://esm-protein-atlas/v1/`,`--no-sign-request`,**完全免费,无 AWS 账户,不走 Athena**。
> 2. **SAE features 实际规模 306 TB,不是 870 GB**!原始数据包含 per-residue + per-protein 特征向量,需要预处理才能压到 ~870 GB 倒排索引。
> 3. **关键新数据集**:`HMM Results` (653 MB,含 6.8B 蛋白 Pfam/taxonomy)、`Normalization` (192 KB,IPF 归一化)、`SAE Clusters` (26 GB,7.7M 簇成员关系)。
> 4. **真正的预算瓶颈是存储,不是算力**;推荐路径 Y(7.7M representative 索引)而非路径 X(全量 6.8B 索引)。

### 11.1 数据规模与存储(v4.3 基于官方数据)

#### 11.1.1 ESM Atlas 官方数据集(从 S3 公开桶免费拉取)

| 数据集 | 实际大小 | 内容 | 用途 |
|---|---|---|---|
| **HMM Results** | **653 MB** | 6.8B 蛋白 Pfam + taxonomy 注释 | **必下**——评估 ground truth |
| **Normalization** | **192 KB** | SAE feature 归一化(max_idf_log10) | **必下**——IPF 加权 |
| **SAE Clusters** | **26.0 GB** | 708M non-singleton clusters + 7.7M ≥50 | **必下**——聚类关系 + representative |
| **Protein_to_accession** | 162 GB | protein ID → UniProt accession | 查 ID 映射 |
| **Sequences** | **2.20 TB** | 6.8B 蛋白序列 | 序列信息 |
| **SAE features** | **306 TB** | per-protein + per-residue feature vectors | **原始数据,路径 Y 不全下** |
| **Structures** | 68.9 TB | 1B 蛋白结构 | P3+ 用 |
| **All Data** | 377 TB | 完整集合 | 不需要 |

**下载命令**(完全免费,无 AWS 账户):
```bash
aws s3 sync --no-sign-request s3://esm-protein-atlas/v1/clusters/data/representative_proteins.parquet /mydrive
aws s3 sync --no-sign-request s3://esm-protein-atlas/v1/clusters/indexes/secondary/cluster_members/ /mydrive
aws s3 cp --no-sign-request s3://esm-protein-atlas/v1/normalization/max_idf_log10.pkl /mydrive/
```

#### 11.1.2 倒排索引本身的规模(算术)

| 维度 | 570K SwissProt | **路径 Y:7.7M representative** | **路径 X:6.8B 全量** |
|---|---|---|---|
| 主数据(每蛋白 64 features × 2 bytes int16) | ~73 MB | **~1 GB** | ~870 GB |
| Posting list(per-feature → protein_id) | ~140 MB | **~2 GB** | ~1.76 TB |
| ES 元数据 + 副本(单副本 30%) | ~300 MB | **~3-5 GB** | **~3.4 TB(单副本)** |
| 生产(2 副本) | ~600 MB | **~10 GB** | ~6.8 TB |
| **物理存储需求** | 单 SSD | **1 TB SSD 单机够** | **300+ TB 阵列** |
| 内存 | 4 GB | 16-32 GB | 64-128 GB |

**关键洞察**:路径 Y 用 representative 替代全量,**存储从 300 TB 降到 1 TB**——可单机跑。Expand 到全 cluster 后仍能挖暗物质(7.7M 簇 × 平均 100 蛋白/cluster ≈ 770M 蛋白,覆盖 ~10% 宇宙,但覆盖度高置信结构)。

### 11.2 算力与网络(v4.3)

- **SAE 推理**:**不需要 GPU**——所有 SAE features 已在 ESM Atlas S3 公开桶
- **下载方式**:**`aws s3 sync --no-sign-request s3://esm-protein-atlas/v1/...`**——**完全免费,无 AWS 账户,无 Athena**
- **下载量(按路径选择)**:
  - **路径 Y(7.7M representative)**:HMM(653 MB) + Clusters(26 GB) + Normalization(192 KB) + representative proteins 的 SAE features 提取(从 306 TB 桶里精准过滤,约 3-5 GB)+ accession(162 GB,可选)
    - **总下载:~190 GB**(几乎全是 accession 映射)
    - **网络时间**:1 Gbps 带宽 ~25 分钟
  - **路径 X(6.8B 全量)**:Sequences(2.2 TB) + SAE features(306 TB)= **~308 TB**
    - **网络时间**:1 Gbps 带宽 ~27 天;10 Gbps 专线 ~3 天
- **ES 索引构建**:
  - **路径 Y**:1 GB 主数据 / 节点,~10 分钟单节点
  - **路径 X**:870 GB 主数据 / 节点,~3 小时单节点
- **预处理(路径 X 必须)**:306 TB → 870 GB 主数据,需 Spark 集群跑 1-3 天
- **本地存储需求**:
  - 路径 Y:**1 TB SSD**(~1.5 万 RMB)
  - 路径 X:**300 TB SSD 阵列**(~30-50 万 RMB)

### 11.3 成本估算(v4.3 路径 Y 推荐)

> **v4.3 关键修正**:
> - 之前方案 A/B/C 的"Athena 收费"部分**已作废**——`aws s3 sync --no-sign-request` 完全免费,**Athena 不必用**。
> - 之前方案 A/B/C 的"6.8B 全量存储 30 TB"严重低估了实际数据规模(SAE features 原始 306 TB)。
> - 真正推荐路径 Y(7.7M representative 索引),存储只需 1 TB SSD,单机 1.5 万 RMB 跑得动。

#### 路径 Y 成本(7.7M representative,**强烈推荐**)

| 项目 | 规格 | 费用 |
|---|---|---|
| ESM Atlas 数据下载 | `aws s3 sync --no-sign-request`,~190 GB(元数据) + 几 GB representative features | **$0(完全免费,无 AWS 账户)** |
| 本地物理机 | 64 核 + 128GB RAM + **1 TB NVMe SSD** | **~1.5 万 RMB(一次性)** |
| 一次性投入 | | **~1.5 万 RMB** |
| 持续 | 电费 | ~¥150/月 |
| **1 年总成本** | | **~1.7 万** |
| **5 年总成本** | | **~2.4 万** |

#### 路径 X 成本(全量 6.8B,财大气粗)

| 项目 | 规格 | 费用 |
|---|---|---|
| ESM Atlas 数据下载 | 308 TB 走 `aws s3 sync --no-sign-request` | **$0(完全免费)** |
| 预处理(306 TB → 870 GB) | Spark/Hadoop 集群临时算力 | **~3-5 万(临时云)** |
| 物理机 | 64 核 + 256GB RAM + **300 TB SSD 阵列** | **~30-50 万 RMB** |
| 一次性投入 | | **~35-55 万 RMB** |
| **1 年总成本** | | **~36-56 万** |

#### 路径 A 成本(570K SwissProt 笔记本版)

| 项目 | 规格 | 费用 |
|---|---|---|
| 数据下载 | HMM Results 653 MB + 570K 蛋白 accession | **$0** |
| 本地笔记本 | 自有笔记本 | **0** |
| P4 GPU(算 PETase 激活,半天) | 阿里云 A100 | **~1000 RMB** |
| **总预算** | | **< 2000 RMB** |

#### 路径对比(2026 年一次性投入)

| 路径 | 数据规模 | 暗物质 | 物理机成本 | 总预算 | 1 年持续 |
|---|---|---|---|---|---|
| **A. 570K 笔记本** | 0.05% | ❌ | 0(自有) | **< 2K** | ~1000(GPU) |
| **Y. 7.7M representative**(推荐) | 0.1% 蛋白但覆盖 10-20% 宇宙(expand 后) | ✅(via cluster) | **1.5 万** | **~1.7 万** | ~2000(电) |
| **X. 6.8B 全量** | 100% | ✅(直接) | 30-50 万 | **~50 万** | ~5000(电) |

**强烈推荐路径 Y**:
- 一次性 1.5 万(1 TB SSD 桌面服务器),后续只付电费
- Expand 到 cluster 后,覆盖 7.7M × 平均 100 蛋白/cluster ≈ 770M 蛋白(占全宇宙 ~11% 但覆盖高置信结构)
- 暗物质挖掘:**完全够用**——暗物质蛋白往往没有代表性,走"无 Pfam 约束的 cluster 成员"反而能挖到
- 与路径 A(< 2K)比,贵 8 倍;与路径 X(50 万)比,便宜 300 倍

### 11.4 时间线(6.8B 版)

| 阶段 | 时间 | 关键交付 |
|---|---|---|
| **P1a** SwissProt 验证 | 第 1-2 周 | 570K 版 PoC,验证管线、字段、查询模式 |
| **P1b** 6.8B 全量扩展 | 第 3-6 周 | 3 节点 ES 集群上线,6.8B 文档灌库,性能基线 |
| **P2** 暗物质专项优化 | 第 7-10 周 | IPF 过滤、posting list 压缩、暗物质查询响应 < 1s |
| **P3** 评估(含暗物质) | 第 11-18 周 | 8 大类别 + 暗物质专属指标(`dark_matter_ratio`) |
| **P4** 案例(塑料降解酶 + 暗物质版) | 第 19-30 周 | 6.8B 全量查询、远缘同源体、wet lab 候选 |
| **P5** 论文开源 | 第 31-42 周 | 投稿、release |

**总周期**:**~10 个月**(vs 原 v4 9 个月,多 1 个月用于 6.8B 灌库)

### 11.5 关键技术挑战

1. **Athena JOIN 6.8B 行的稳定性**
   - 全表 JOIN 可能超时(S3 manifest 巨大)
   - **应对**:分批(按 cluster_id 哈希分片,每批 100M 行)+ 本地 merge
   - 或用 Athena CREATE TABLE AS SELECT 物化中间结果

2. **6.8B 文档的 ES 集群规模**
   - 单节点 ES 在 ~1B 文档以下较稳定;6.8B 强烈建议 3 节点起步
   - 分片策略:`shard_size = 30-50 GB`,6.8B / 30 GB ≈ 230 个分片 → 3 节点 × 80 分片/节点
   - 副本数:1 副本(开发)/ 2 副本(生产)

3. **暗物质查询的"过召回"问题**
   - 6.8B 库下,弱约束查询可能返回 ~百万级结果
   - **应对**:
     - 默认 must ≥ 3 个 feature
     - 引入 IDF 加权(`boost` 字段)
     - 暗物质专项:必须用严格 must 才能进入"暗物质候选名单"

4. **内存压力**
   - 6.8B × 64 features = 440B 个 (protein, feature) 对
   - ES JVM heap 不存原始数据(用 doc_values),但 fielddata 可能爆
   - **应对**:`indices.fielddata.cache.size` 限制 + `keyword` 数组而非 `integer`

### 11.6 暗物质查询专属评估指标

| 指标 | 含义 | 目标 |
|---|---|---|
| `uncharacterized_ratio` | 命中结果中"无 Pfam 注释"的比例 | > 30%(说明挖到暗物质) |
| `cluster_novelty` | 命中结果中"无 cluster representative 同源"的比例 | > 20% |
| `precision_dark` | 在随机抽样 100 个结果中,wet lab / 文献确认的"真暗物质酶"比例 | > 5% |
| `recall_known_dark` | 已知"功能性被重新发现"的暗物质蛋白被召回到 top-1000 的比例 | > 50% |

### 11.7 修订后的总预算(v4.3 三种粒度)

| 粒度 | 一次性 | 持续(月) | 1 年总成本 | 暗物质 |
|---|---|---|---|---|
| **A. 570K 笔记本版** | < 2K | 0 | **< 2K** | ❌ |
| **Y. 7.7M representative 自建 1TB SSD** | **1.5 万** | ¥150 电费 | **~1.7 万** | ✅(via cluster expand) |
| **X. 6.8B 全量 300TB SSD 阵列** | 30-50 万 + 3-5 万预处理 | ¥500 电费 | **~36-56 万** | ✅(直接) |

**强烈推荐路径 Y**:1.5 万一次性投入 + 月电费 ¥150。1 年总成本 **~1.7 万**,3-5 年长期持有综合成本最低。

对比 v4 的 < 1000 RMB:贵了一个数量级,但**换来了暗物质叙事的真实落地**——这是值得的。计算平权的核心是"**不必拥有 H100 集群也能做大模型驱动的科学发现**",1.5 万 RMB 一次性投入在生物学/AI 交叉课题里属于**最低成本量级**。

### 11.8 修订建议汇总(v4.3)

1. **P1 升级为两阶段**:P1a(2 周,570K)+ P1b(2 周,**路径 Y 7.7M representative**)
2. **P4 案例增加暗物质版**:在 7.7M representative + cluster expand 上跑"无 Pfam 约束"的塑料降解酶查询
3. **P3 评估增加 §11.6 的暗物质专属指标**
4. **预算从 < 1000 RMB 升级为 1.5 万一次性(路径 Y 1TB SSD 自建)**,与 v3 的 35-80 万相比仍节省 95%+
5. **数据获取方式从"Athena JOIN"改为"`aws s3 sync --no-sign-request`"**——**完全免费,无 AWS 账户**
6. **下载数据集清单**:
   - 必下:HMM Results(653 MB)、Normalization(192 KB)、SAE Clusters(26 GB)
   - 选下:Sequences(2.20 TB,部分)、Protein_to_accession(162 GB)
   - 不下(首期):SAE features(306 TB,预处理后才用上)、Structures(68.9 TB)
7. **创新点第 2 条措辞调整**:从"零 GPU"改为"**零本地 GPU,自建 1TB SSD 桌面服务器跑 ES 索引**"——更准确

---

**v4 → v4.1 → v4.2 → v4.3 关键差异**:
- **v4.2 → v4.3(根本性升级)**:Atlas 数据从"Athena JOIN"改为"`aws s3 sync --no-sign-request`"(免费);SAE features 实际规模 306 TB(不是 870 GB);新增 `HMM Results`(653 MB)、`Normalization`(192 KB)等关键数据集
- **v4.1 → v4.2**:主数据从 3.5 TB 修正为 870 GB;物理存储从 30 TB 修正为 8 TB;自建预算从 5-8 万修正为 3-4 万
- **v4 → v4.1**:暗物质从 "future work" 升为"首期核心 case";数据规模从 570K 升到 6.8B
- **最终推荐路径**:**Y(7.7M representative + cluster expand),1.5 万一次性,1 年 1.7 万**——比之前又省 50%+
- 算力约束:依然**零本地 GPU**;存储是真正瓶颈,但路径 Y 1TB SSD 单机搞定
- **修正历史**:
  - Athena **不在 Free Tier**,按 $5/TB 收费
  - 但 v4.3 进一步发现:**Athena 完全不需要用**——`aws s3 sync --no-sign-request` 走 Open Data Sponsorship 完全免费
  - ESM Atlas 数据本身免费(AWS Open Data Sponsorship)
