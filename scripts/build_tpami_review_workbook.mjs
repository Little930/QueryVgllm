import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outputDir = path.join(root, "outputs", "tpami_review");
const sourcePath = path.join(outputDir, "source_rows.json");
const outputPath = path.join(outputDir, "TPAMI_综述文献分类整理.xlsx");

const rows = JSON.parse(await fs.readFile(sourcePath, "utf8"));

const mainSections = [
  {
    id: "00",
    name: "视觉语言基础与语言到动作桥接",
    thesis: "用 CLIP/BLIP/Flamingo/GPT-4V/LLaVA/SayCan/VoxPoser 等交代 VLM 能力如何进入机器人任务规划、affordance grounding 和可组合控制。",
    dimensions: "预训练来源；空间/动作 grounding；是否闭环；是否真机验证",
  },
  {
    id: "01",
    name: "相关综述与问题定义",
    thesis: "用已有 VLA、数据集、world model 和具身 AI 综述定位本文差异：从数据-任务-评测的角度组织 mobile manipulation 与 generalist robots。",
    dimensions: "综述范围；是否覆盖数据引擎；是否覆盖 mobile manipulation；是否讨论评价指标",
  },
  {
    id: "02",
    name: "仿真环境与任务世界",
    thesis: "说明 embodied AI 从导航仿真器走向可交互、物理真实、家庭活动级任务世界，是后续数据引擎和 benchmark 的基础。",
    dimensions: "物理真实性；可交互物体；家庭活动覆盖；渲染/并行效率；sim-to-real 路径",
  },
  {
    id: "03",
    name: "真实机器人数据与采集",
    thesis: "围绕真实数据规模化的瓶颈展开：低成本遥操作、跨本体合并、in-the-wild 采集、Ego/human video 与数据格式统一。",
    dimensions: "机器人本体；采集接口；规模；模态；任务覆盖；可复现性",
  },
  {
    id: "04",
    name: "数据引擎与合成扩展",
    thesis: "讨论如何突破真机数据成本：生成式仿真、real-to-sim、渲染增强、自动轨迹生成、合成动作数据与 scaling law。",
    dimensions: "生成源；是否需要动力学；合成规模；真实性；泛化收益；sim-to-real 证据",
  },
  {
    id: "05",
    name: "移动操作和开放世界任务",
    thesis: "把 mobile manipulation 作为 TPAMI 综述的主 stress test：导航、定位、抓取、长程规划、失败恢复和开放词汇理解必须联合评价。",
    dimensions: "Nav+Manip 耦合；长程任务；开放词汇；失败恢复；真实/仿真闭环；评价指标",
  },
  {
    id: "06",
    name: "灵巧/双臂/全身操作数据",
    thesis: "说明 tabletop 单臂 VLA 之外的下一层复杂度：灵巧手、双臂、全身/人形操作带来动作空间、接触和本体迁移问题。",
    dimensions: "动作维度；接触丰富程度；双臂/全身协调；对象复杂度；安全与遥操作难度",
  },
  {
    id: "07",
    name: "VLA与Robot Brain模型",
    thesis: "梳理从 embodied VLM 到 VLA/robot brain 的模型路线：web knowledge transfer、视频预训练、mid-training、action tokenization、多模态导航与具身推理。",
    dimensions: "视觉编码器；语言模型；动作表示；训练阶段；数据混合；泛化评测",
  },
  {
    id: "08",
    name: "评测、推理与失效分析",
    thesis: "强调只看 success rate 不够，TPAMI 综述应系统比较 task progress、re-planning、空间推理、RoboVQA/EQA、幻觉和失败类型。",
    dimensions: "指标粒度；是否支持部分进展；是否评估推理幻觉；失败标签；benchmark 难度",
  },
  {
    id: "09",
    name: "相邻领域和背景参考",
    thesis: "放置自动驾驶、3D 生成、世界模型等相邻领域材料，用作方法论类比，不应喧宾夺主。",
    dimensions: "可迁移的数据治理经验；annotation quality；world model 训练范式；3D asset 生成",
  },
];

const sectionByName = Object.fromEntries(mainSections.map((s) => [s.name, s]));

const originalHeaders = [
  "序号",
  "论文名称",
  "链接",
  "Publication",
  "Type",
  "简要概括",
  "数据量",
  "评价指标",
  "发布时间",
  "分级",
];

const organizedHeaders = [
  "原序号",
  "论文名称",
  "链接",
  "Publication",
  "年份",
  "原Type",
  "原分级",
  "引用优先级",
  "证据强度",
  "TPAMI主章节",
  "二级主题",
  "贡献类型",
  "研究对象",
  "数据来源",
  "机器人形态",
  "任务尺度",
  "评测关注点",
  "写作用途",
  "建议放置",
  "核心论点提炼",
  "数据量",
  "评价指标",
  "简要概括",
  "发布时间",
  "核查标记",
];

const priorityOrder = { "P0 必引主线": 0, "P1 重点支撑": 1, "P2 背景比较": 2, "P3 可选/边缘": 3 };

function compact(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function textOf(row) {
  return [
    row["论文名称"],
    row["Publication"],
    row["Type"],
    row["简要概括"],
    row["数据量"],
    row["评价指标"],
    row["分级"],
  ]
    .map(compact)
    .join(" ")
    .toLowerCase();
}

function hasAny(text, words) {
  return words.some((w) => text.includes(w.toLowerCase()));
}

function yearOf(row) {
  const date = compact(row["发布时间"]);
  const pub = compact(row["Publication"]);
  const title = compact(row["论文名称"]);
  const all = `${date} ${pub} ${title}`;
  let m = all.match(/(20\d{2})/);
  if (m) return Number(m[1]);
  m = all.match(/\b(ICCV|CVPR|ECCV|ICRA|IROS|RSS|CoRL|NeurIPS|ICLR|ICML|IJRR)\s*'?(\d{2})\b/i);
  if (m) return 2000 + Number(m[2]);
  return "";
}

function evidenceOf(row) {
  const pub = compact(row["Publication"]);
  const t = `${pub} ${compact(row["Type"])} ${compact(row["论文名称"])}`.toLowerCase();
  if (hasAny(t, ["在投", "under review"])) return "在投/待核查";
  if (hasAny(t, ["blog", "system card", "technical report", "report", "开源项目"])) return "技术报告/博客/项目";
  if (hasAny(t, ["researchgate", "preprint"])) return "预印本/低可验证";
  if (hasAny(t, ["arxiv"])) return "arXiv/预印本";
  if (pub) return "已发表会议/期刊";
  return "未知/待补";
}

function priorityOf(row, section, contribution) {
  const level = compact(row["分级"]);
  const title = compact(row["论文名称"]).toLowerCase();
  if (["必引", "核心前沿", "核心补充"].includes(level)) return "P0 必引主线";
  if (
    hasAny(title, [
      "rt-2",
      "open x-embodiment",
      "droid",
      "mobile aloha",
      "homerobot",
      "ok-robot",
      "gr-2",
      "π0.5",
      "mobilemanibench",
      "emmoe",
      "lambda",
      "air oa",
      "airoa",
    ])
  ) {
    return "P0 必引主线";
  }
  if (["重要", "Insight"].includes(level)) return "P1 重点支撑";
  if (["可选", "marginal"].includes(level)) return "P3 可选/边缘";
  if (!level) return "P2 背景比较";
  if (hasAny(level, ["借鉴", "前沿借鉴"])) return "P2 背景比较";
  if (contribution === "Survey/Meta-analysis" && section !== "09_相邻领域和背景参考") return "P1 重点支撑";
  return "P2 背景比较";
}

function contributionOf(row) {
  const title = compact(row["论文名称"]).toLowerCase();
  const type = compact(row["Type"]).toLowerCase();
  const pub = compact(row["Publication"]).toLowerCase();
  const core = `${title} ${type} ${pub}`;
  const t = textOf(row);
  if (hasAny(core, ["survey", "综述", "review", "meta-analytic"])) return "Survey/Meta-analysis";
  if (hasAny(core, ["system card", "blog", "technical report", "technical report"])) return "Technical report";
  if (
    hasAny(core, [
      "vla",
      "vision-language-action",
      "robot brain",
      "机器人脑模型",
      "robotic foundation",
      "foundation model",
      "gemini robotics",
      "rt-2",
      "palm-e",
      "π0",
      "pi0",
      "gr-2",
      "vlm",
      "图文对齐",
      "语言机器人对齐",
      "视频规划",
      "空间推理",
      "导航 vlm",
      "vlm到vla",
      "具身推理vla",
      "domino",
      "embodiedmidtrain",
    ])
  ) {
    return "Model/Robot foundation model";
  }
  if (hasAny(core, ["robovqa", "eqa", "reasoning", "推理评测", "具身问答"])) return "Evaluation/QA benchmark";
  if (hasAny(core, ["data engine", "数据引擎", "synthetic", "合成", "generative simulation", "自动数据", "real-to-sim", "render", "scaling", "scale"])) return "Data engine / synthetic scaling";
  if (hasAny(core, ["benchmark", "bench", "评测"])) return "Benchmark";
  if (hasAny(core, ["dataset", "data set", "数据集", "demonstrations", "真机数据", "真实双臂数据", "人形数据", "人类 ego"])) return "Dataset";
  if (hasAny(core, ["simulator", "simulation", "仿真", "physics engine", "environment", "平台", "isaac", "habitat", "sapien", "genesis", "omnigibson"])) return "Simulator/Environment";
  if (hasAny(core, ["teleoperation", "teleop", "遥操作", "hardware", "interface", "采集硬件"])) return "Collection interface/hardware";
  if (hasAny(core, ["planner", "planning", "policy", "策略", "trajectory", "轨迹"])) return "Method/Policy";
  if (hasAny(title, ["ok-robot", "spot-compose", "homie", "suite"])) return "System";
  if (hasAny(t, ["dataset", "数据集"])) return "Dataset";
  if (hasAny(t, ["benchmark", "评测"])) return "Benchmark";
  return "Method/System";
}

function sectionOf(row, contribution) {
  const t = textOf(row);
  const title = compact(row["论文名称"]).toLowerCase();
  const type = compact(row["Type"]).toLowerCase();
  const core = `${title} ${type}`;
  if (hasAny(core, ["自驾", "autonomous driving", "3d generation", "world model"]) && !hasAny(core, ["robotic simulation", "robotics"])) return "09_相邻领域和背景参考";
  if (contribution === "Survey/Meta-analysis") return "01_相关综述与问题定义";
  if (
    hasAny(core, ["clip", "blip", "flamingo", "llava", "gpt-4v", "visual instruction", "saycan", "do as i can", "voxposer", "unipi", "spatialvlm", "图文对齐", "语言机器人对齐", "视频规划", "vlm规划器", "空间推理 vlm"])
  ) {
    return "00_视觉语言基础与语言到动作桥接";
  }
  if (type === "vlm" || hasAny(core, ["gpt", "visual language model"])) return "00_视觉语言基础与语言到动作桥接";
  if (
    hasAny(core, [
      "rt-2",
      "palm-e",
      "gr-2",
      "humanvla",
      "robobrain",
      "π0",
      "pi0",
      "vlm4vla",
      "gemini robotics",
      "wholebodyvla",
      "omni-vla",
      "omnivla",
      "domino",
      "embodiedmidtrain",
      "vla",
      "vision-language-action",
      "robot brain",
      "机器人脑模型",
      "具身视觉语言基础模型",
      "开放世界vla",
      "多模态导航 vla",
      "具身推理vla",
      "人形 loco-mani",
    ])
  ) {
    return "07_VLA与Robot Brain模型";
  }
  if (hasAny(core, ["robovqa", "eqa", "question answering", "reasoning", "推理评测", "具身问答", "illusion"]) || (contribution === "Evaluation/QA benchmark")) return "08_评测、推理与失效分析";
  if (hasAny(core, ["mobile manipulation", "移动操作", "nav+manip", "navigation + manipulation", "open-vocabulary mobile", "ovmm", "arm point nav", "armpointnav", "household nav"]) || hasAny(title, ["home robot", "home-robot", "ok-robot", "emmoe", "lambda", "mobilemanibench", "harmonic mobile", "mobile aloha"])) {
    return "05_移动操作和开放世界任务";
  }
  if (hasAny(core, ["dex", "灵巧", "bimanual", "双臂", "hand", "grasp", "in-hand", "handover", "humanoid", "人形", "wholebody", "whole-body", "loco-manip", "aloha"])) {
    return "06_灵巧/双臂/全身操作数据";
  }
  if (hasAny(core, ["data engine", "数据引擎", "synthetic", "合成", "generative simulation", "real-to-sim", "render", "scaling law", "scale", "sim2real", "zero robot", "human videos", "egocentric video"])) return "04_数据引擎与合成扩展";
  if (hasAny(core, ["simulator", "simulation", "仿真", "physics engine", "environment", "平台", "isaac", "habitat", "sapien", "behavior", "robocasa", "maniskill", "genesis", "omnigibson", "softbody", "garment"])) return "02_仿真环境与任务世界";
  if (hasAny(core, ["dataset", "数据集", "teleoperation", "teleop", "遥操作", "in-the-wild", "true robot", "真实机器人", "真机", "采集硬件", "open x-embodiment", "droid", "bridge", "rh20t", "agibot"])) return "03_真实机器人数据与采集";
  if (hasAny(t, ["failure", "失效", "幻觉", "metric"])) return "08_评测、推理与失效分析";
  return "03_真实机器人数据与采集";
}

function subtopicOf(row, section) {
  const t = textOf(row);
  if (hasAny(t, ["failure", "失败", "re-plan", "replan"])) return "失败恢复与错误分析";
  if (hasAny(t, ["open-vocabulary", "开放词汇", "open-world", "开放世界"])) return "开放词汇/开放世界";
  if (hasAny(t, ["navigation", "vln", "objectnav", "导航"])) return "导航与语言条件移动";
  if (hasAny(t, ["mobile manipulation", "移动操作", "nav+manip"])) return "移动操作/Nav+Manip";
  if (hasAny(t, ["household", "home", "everyday", "家庭", "日常"])) return "家庭日常任务";
  if (hasAny(t, ["dexterous", "dex", "灵巧", "in-hand"])) return "灵巧手与接触丰富操作";
  if (hasAny(t, ["bimanual", "双臂", "aloha"])) return "双臂/低成本遥操作";
  if (hasAny(t, ["humanoid", "人形", "whole-body", "loco-manip"])) return "人形/全身操作";
  if (hasAny(t, ["sim2real", "real-to-sim", "render", "synthetic", "合成"])) return "合成数据与Sim2Real";
  if (hasAny(t, ["scaling", "scale", "1b", "billion"])) return "数据规模化与Scaling Law";
  if (hasAny(t, ["robovqa", "eqa", "question", "reasoning", "spatial", "推理"])) return "具身问答/空间推理";
  if (hasAny(t, ["simulator", "仿真", "physics engine", "environment"])) return "仿真平台/交互环境";
  if (hasAny(t, ["vla", "vision-language-action", "robot brain"])) return "VLA/Robot Brain";
  if (hasAny(t, ["vlm", "clip", "blip", "flamingo", "llava", "gpt-4v"])) return "VLM基础模型";
  return sectionByName[section.replace(/^\d+_/, "")]?.name ?? "通用支撑";
}

function researchObjectOf(row) {
  const t = textOf(row);
  const items = [];
  if (hasAny(t, ["mobile manipulation", "移动操作", "nav+manip", "ovmm"])) items.push("Mobile manipulation");
  if (hasAny(t, ["navigation", "vln", "objectnav", "导航"])) items.push("Embodied navigation");
  if (hasAny(t, ["dex", "灵巧", "hand", "grasp", "in-hand"])) items.push("Dexterous manipulation");
  if (hasAny(t, ["bimanual", "双臂", "aloha"])) items.push("Bimanual manipulation");
  if (hasAny(t, ["humanoid", "人形", "whole-body", "loco-manip"])) items.push("Humanoid/whole-body");
  if (hasAny(t, ["vla", "vlm", "robot brain", "foundation model"])) items.push("VLM/VLA foundation");
  if (hasAny(t, ["simulator", "simulation", "仿真", "environment", "physics engine"])) items.push("Simulation world");
  if (hasAny(t, ["robovqa", "eqa", "question", "reasoning", "推理"])) items.push("Embodied reasoning");
  if (hasAny(t, ["autonomous driving", "自驾"])) items.push("Adjacent: autonomous driving");
  return items.length ? [...new Set(items)].join(" / ") : "General robotic manipulation";
}

function dataSourceOf(row) {
  const t = textOf(row);
  const sources = [];
  if (hasAny(t, ["真实", "真机", "real-world", "real robot", "in-the-wild", "human support robot", "spot", "fetch robot", "agibot"])) sources.push("真实机器人");
  if (hasAny(t, ["simulation", "simulator", "仿真", "isaac", "habitat", "sapien", "omnigibson", "robocasa", "maniskill", "genesis"])) sources.push("仿真");
  if (hasAny(t, ["web-scale", "web knowledge", "web", "internet"])) sources.push("Web/互联网");
  if (hasAny(t, ["human video", "egocentric", "ego", "人类视频", "human-to-robot"])) sources.push("人类视频/Ego");
  if (hasAny(t, ["synthetic", "合成", "generative", "render", "real-to-sim", "zero robot"])) sources.push("合成/生成");
  if (hasAny(t, ["survey", "综述", "review"])) sources.push("文献综述");
  return sources.length ? [...new Set(sources)].join(" + ") : "未明确";
}

function embodimentOf(row) {
  const t = textOf(row);
  const items = [];
  if (hasAny(t, ["mobile", "移动", "base", "底盘", "hsr", "fetch robot", "spot"])) items.push("移动底盘+机械臂");
  if (hasAny(t, ["bimanual", "双臂", "aloha"])) items.push("双臂");
  if (hasAny(t, ["dex", "灵巧", "hand", "gripper", "grasp"])) items.push("灵巧手/夹爪");
  if (hasAny(t, ["humanoid", "人形", "whole-body", "loco"])) items.push("人形/全身");
  if (hasAny(t, ["cross-embodiment", "跨本体", "x-embodiment"])) items.push("跨本体");
  if (hasAny(t, ["navigation", "vln", "objectnav"])) items.push("导航智能体");
  return items.length ? [...new Set(items)].join(" + ") : "通用/未限定";
}

function taskScaleOf(row) {
  const t = textOf(row);
  if (hasAny(t, ["long-horizon", "long horizon", "长程", "everyday", "household", "日常", "家庭"])) return "长程家庭任务";
  if (hasAny(t, ["open-ended", "open-world", "开放世界", "开放环境"])) return "开放环境任务";
  if (hasAny(t, ["grasp", "grasping", "in-hand", "handover", "抓取"])) return "抓取/接触技能";
  if (hasAny(t, ["navigation", "vln", "objectnav", "导航"])) return "导航/目标定位";
  if (hasAny(t, ["question answering", "robovqa", "eqa", "reasoning"])) return "具身问答/推理";
  if (hasAny(t, ["teleoperation", "采集", "dataset", "数据集"])) return "数据采集/模仿学习";
  return "通用操作/系统能力";
}

function evalFocusOf(row) {
  const t = textOf(row);
  const f = [];
  if (hasAny(t, ["success rate", "sr", "成功率", "success"])) f.push("Success Rate");
  if (hasAny(t, ["task progress", "tp", "progress"])) f.push("Task Progress");
  if (hasAny(t, ["re-plan", "replan", "srr", "失败", "failure"])) f.push("失败恢复/Re-plan");
  if (hasAny(t, ["generalization", "unseen", "泛化"])) f.push("泛化");
  if (hasAny(t, ["sim2real", "real-to-sim"])) f.push("Sim2Real");
  if (hasAny(t, ["spatial", "空间"])) f.push("空间推理");
  if (hasAny(t, ["scaling", "scale"])) f.push("Scaling");
  if (hasAny(t, ["efficiency", "效率"])) f.push("效率");
  if (hasAny(t, ["vqa", "eqa", "question"])) f.push("QA/Reasoning");
  return f.length ? [...new Set(f)].join(" / ") : "规模、覆盖度、任务成功与可复现性";
}

function writingUse(priority, contribution, section) {
  if (priority.startsWith("P0")) return "主线必引；适合放章节开头、核心对比表和未来方向";
  if (priority.startsWith("P1")) return "重点支撑；用于论证趋势、补充 benchmark/model/data 证据";
  if (priority.startsWith("P3")) return "可选引用；仅在细分主题或补充材料中使用";
  if (contribution === "Survey/Meta-analysis") return "Related Work 背景；用于说明本文与既有综述差异";
  if (section.startsWith("09")) return "方法论类比；不建议进入主线大篇幅";
  return "背景比较；用于补充脉络、扩展表格或脚注";
}

function placement(section, subtopic) {
  return `${section} / ${subtopic}`;
}

function claimOf(row, contribution, section) {
  const summary = compact(row["简要概括"]);
  const title = compact(row["论文名称"]);
  const lead = summary ? summary.slice(0, 90) : title;
  if (contribution === "Dataset") return `作为数据集证据：比较规模、模态、采集方式和任务覆盖；可引出数据稀缺与跨本体泛化问题。原表摘要：${lead}`;
  if (contribution === "Benchmark") return `作为 benchmark 证据：比较任务定义、指标粒度和是否支持开放词汇/长程/失败恢复。原表摘要：${lead}`;
  if (contribution === "Simulator/Environment") return `作为仿真环境证据：说明可交互物理世界、家庭活动和可扩展采样如何支撑 embodied robot learning。原表摘要：${lead}`;
  if (contribution === "Data engine / synthetic scaling") return `作为数据扩展证据：讨论合成数据、real-to-sim、渲染增强或自动轨迹生成对 scaling 的贡献与风险。原表摘要：${lead}`;
  if (contribution === "Model/Robot foundation model") return `作为模型路线证据：比较 VLM/VLA 架构、动作表示、训练数据混合和开放世界泛化。原表摘要：${lead}`;
  if (contribution === "Survey/Meta-analysis") return `作为综述参照：用于定位本文 novelty，避免与已有 VLA/数据/世界模型综述重复。原表摘要：${lead}`;
  return `作为系统/方法证据：抽取其核心假设、任务接口和实验设置，用于支撑章节论点。原表摘要：${lead}`;
}

function checkFlags(row, year, evidence) {
  const flags = [];
  if (!compact(row["论文名称"])) flags.push("缺论文名");
  if (!compact(row["链接"])) flags.push("缺链接");
  if (!compact(row["Publication"])) flags.push("缺Publication");
  if (!compact(row["Type"])) flags.push("缺Type");
  if (!compact(row["分级"])) flags.push("缺分级");
  if (!year) flags.push("缺年份");
  if (evidence.includes("待核查") || evidence.includes("博客") || evidence.includes("低可验证")) flags.push(evidence);
  const pub = compact(row["Publication"]);
  if (/arxiv/i.test(pub) && !/(20\d{2})/.test(pub + compact(row["发布时间"]))) flags.push("arXiv年份待补");
  return flags.join("；");
}

const organized = rows
  .map((row, idx) => {
    const contribution = contributionOf(row);
    const section = sectionOf(row, contribution);
    const subtopic = subtopicOf(row, section);
    const year = yearOf(row);
    const evidence = evidenceOf(row);
    const priority = priorityOf(row, section, contribution);
    const flags = checkFlags(row, year, evidence);
    return {
      "原序号": row["序号"] ?? idx + 1,
      "论文名称": compact(row["论文名称"]),
      "链接": compact(row["链接"]),
      "Publication": compact(row["Publication"]),
      "年份": year,
      "原Type": compact(row["Type"]),
      "原分级": compact(row["分级"]),
      "引用优先级": priority,
      "证据强度": evidence,
      "TPAMI主章节": section,
      "二级主题": subtopic,
      "贡献类型": contribution,
      "研究对象": researchObjectOf(row),
      "数据来源": dataSourceOf(row),
      "机器人形态": embodimentOf(row),
      "任务尺度": taskScaleOf(row),
      "评测关注点": evalFocusOf(row),
      "写作用途": writingUse(priority, contribution, section),
      "建议放置": placement(section, subtopic),
      "核心论点提炼": claimOf(row, contribution, section),
      "数据量": compact(row["数据量"]),
      "评价指标": compact(row["评价指标"]),
      "简要概括": compact(row["简要概括"]),
      "发布时间": compact(row["发布时间"]),
      "核查标记": flags,
      _priorityRank: priorityOrder[priority] ?? 9,
    };
  })
  .sort((a, b) => {
    const s = a["TPAMI主章节"].localeCompare(b["TPAMI主章节"], "zh-Hans-CN");
    if (s) return s;
    if (a._priorityRank !== b._priorityRank) return a._priorityRank - b._priorityRank;
    return (b["年份"] || 0) - (a["年份"] || 0);
  });

const byTitle = new Map();
for (const row of organized) {
  const key = row["论文名称"].toLowerCase();
  if (!key) continue;
  byTitle.set(key, (byTitle.get(key) ?? 0) + 1);
}
for (const row of organized) {
  if (byTitle.get(row["论文名称"].toLowerCase()) > 1) {
    row["核查标记"] = row["核查标记"] ? `${row["核查标记"]}；疑似重复题名` : "疑似重复题名";
  }
}

function countBy(key) {
  const m = new Map();
  for (const r of organized) m.set(r[key], (m.get(r[key]) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0]), "zh-Hans-CN"));
}

function countByYear() {
  const m = new Map();
  for (const r of organized) {
    const y = r["年份"] || "未知";
    m.set(y, (m.get(y) ?? 0) + 1);
  }
  return [...m.entries()].sort((a, b) => String(a[0]).localeCompare(String(b[0])));
}

function topPapers(section, limit = 7) {
  return organized
    .filter((r) => r["TPAMI主章节"] === section)
    .sort((a, b) => a._priorityRank - b._priorityRank || (b["年份"] || 0) - (a["年份"] || 0))
    .slice(0, limit)
    .map((r) => `${r["论文名称"]} (${r["年份"] || "n.d."}, ${r["引用优先级"].split(" ")[0]})`)
    .join("；");
}

function matrix(headers, data) {
  return [headers, ...data.map((row) => headers.map((h) => row[h] ?? ""))];
}

function col(n) {
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function rangeAddress(rowsCount, colsCount) {
  return `A1:${col(colsCount)}${rowsCount}`;
}

function writeSheet(sheet, headers, data, options = {}) {
  const values = matrix(headers, data);
  const address = rangeAddress(values.length, headers.length);
  const range = sheet.getRange(address);
  range.values = values;
  range.format.font = { name: "Microsoft YaHei", size: 10, color: "#111827" };
  range.format.verticalAlignment = "top";
  range.format.wrapText = true;
  range.format.borders = { preset: "inside", style: "thin", color: "#E5E7EB" };
  const header = sheet.getRange(`A1:${col(headers.length)}1`);
  header.format.fill = "#1F4E79";
  header.format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  sheet.freezePanes.freezeRows(1);
  if (options.freezeColumns) sheet.freezePanes.freezeColumns(options.freezeColumns);
  const widths = options.widths ?? {};
  for (let i = 1; i <= headers.length; i += 1) {
    const w = widths[headers[i - 1]] ?? 130;
    sheet.getRange(`${col(i)}1:${col(i)}${values.length}`).format.columnWidthPx = w;
  }
  if (values.length > 1) {
    sheet.getRange(`A2:${col(headers.length)}${values.length}`).format.fill = "#FFFFFF";
  }
  return range;
}

function writeBlock(sheet, startRow, startCol, values, style = {}) {
  const endRow = startRow + values.length - 1;
  const endCol = startCol + values[0].length - 1;
  const address = `${col(startCol)}${startRow}:${col(endCol)}${endRow}`;
  const r = sheet.getRange(address);
  r.values = values;
  r.format = {
    font: { name: "Microsoft YaHei", size: 10, color: "#111827" },
    verticalAlignment: "top",
    wrapText: true,
    ...style,
  };
  return r;
}

function addTitle(sheet, title, subtitle = "") {
  writeBlock(sheet, 1, 1, [[title]], {
    fill: "#1F4E79",
    font: { name: "Microsoft YaHei", size: 16, color: "#FFFFFF", bold: true },
    horizontalAlignment: "left",
  });
  sheet.getRange("A1:H1").merge();
  if (subtitle) {
    writeBlock(sheet, 2, 1, [[subtitle]], {
      fill: "#EAF3F8",
      font: { name: "Microsoft YaHei", size: 10, color: "#1F2937" },
      wrapText: true,
    });
    sheet.getRange("A2:H2").merge();
  }
  sheet.getRange("A1:H2").format.columnWidthPx = 160;
}

const workbook = Workbook.create();

const readme = workbook.worksheets.getOrAdd("README_TPAMI", { renameFirstIfOnlyNewSpreadsheet: true });
addTitle(
  readme,
  "TPAMI综述文献库整理说明",
  "基于原始论文详细表新增分类字段、章节视图、统计汇总和待核查清单；原始数据单独保留，不覆盖源文件。"
);
const readmeRows = [
  ["使用顺序", "先看「统计总览」确认分布，再到「整理总表」筛 P0/P1，最后用「TPAMI写作提纲」搭正文结构。"],
  ["核心筛选", "引用优先级：P0 必引主线、P1 重点支撑、P2 背景比较、P3 可选/边缘。"],
  ["主线建议", "Mobile manipulation 可以作为综述的 stress test，将数据、仿真、VLA、评测和失败恢复串成统一叙事。"],
  ["审稿风险", "arXiv、blog、在投和 ResearchGate 条目已在「证据强度/核查标记」中提示，投稿前建议核对正式版本。"],
  ["维护方式", "新增论文时优先补齐 Publication、Type、数据量、评价指标和分级，再按分类字典选择主章节。"],
];
writeBlock(readme, 4, 1, [["项目", "说明"], ...readmeRows], {
  borders: { preset: "inside", style: "thin", color: "#D1D5DB" },
});
readme.getRange("A4:B4").format.fill = "#1F4E79";
readme.getRange("A4:B4").format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
readme.getRange("A:A").format.columnWidthPx = 140;
readme.getRange("B:B").format.columnWidthPx = 900;
readme.freezePanes.freezeRows(3);

const dict = workbook.worksheets.add("分类字典");
const dictRows = mainSections.map((s) => ({
  "章节编号": s.id,
  "TPAMI主章节": `${s.id}_${s.name}`,
  "章节核心论点": s.thesis,
  "建议比较维度": s.dimensions,
}));
writeSheet(dict, ["章节编号", "TPAMI主章节", "章节核心论点", "建议比较维度"], dictRows, {
  freezeColumns: 2,
  widths: {
    "章节编号": 70,
    "TPAMI主章节": 240,
    "章节核心论点": 620,
    "建议比较维度": 520,
  },
});

const organizedSheet = workbook.worksheets.add("整理总表");
writeSheet(organizedSheet, organizedHeaders, organized, {
  freezeColumns: 2,
  widths: {
    "原序号": 70,
    "论文名称": 360,
    "链接": 260,
    "Publication": 150,
    "年份": 70,
    "原Type": 230,
    "原分级": 95,
    "引用优先级": 115,
    "证据强度": 130,
    "TPAMI主章节": 210,
    "二级主题": 170,
    "贡献类型": 180,
    "研究对象": 190,
    "数据来源": 140,
    "机器人形态": 150,
    "任务尺度": 150,
    "评测关注点": 220,
    "写作用途": 320,
    "建议放置": 320,
    "核心论点提炼": 620,
    "数据量": 520,
    "评价指标": 520,
    "简要概括": 700,
    "发布时间": 130,
    "核查标记": 230,
  },
});
organizedSheet.getRange(`H2:H${organized.length + 1}`).conditionalFormats.add("containsText", {
  text: "P0",
  format: { fill: "#D1FAE5", font: { color: "#065F46", bold: true } },
});
organizedSheet.getRange(`H2:H${organized.length + 1}`).conditionalFormats.add("containsText", {
  text: "P3",
  format: { fill: "#FEE2E2", font: { color: "#991B1B" } },
});
organizedSheet.getRange(`Y2:Y${organized.length + 1}`).conditionalFormats.add("containsText", {
  text: "待",
  format: { fill: "#FEF3C7", font: { color: "#92400E", bold: true } },
});

const grouped = workbook.worksheets.add("按章节分组");
const groupedRows = mainSections.map((s) => {
  const section = `${s.id}_${s.name}`;
  const subset = organized.filter((r) => r["TPAMI主章节"] === section);
  const p0 = subset.filter((r) => r["引用优先级"].startsWith("P0")).length;
  const p1 = subset.filter((r) => r["引用优先级"].startsWith("P1")).length;
  const arxiv = subset.filter((r) => r["证据强度"].includes("arXiv")).length;
  return {
    "TPAMI主章节": section,
    "条目数": subset.length,
    "P0数": p0,
    "P1数": p1,
    "预印本数": arxiv,
    "章节核心论点": s.thesis,
    "建议比较维度": s.dimensions,
    "代表论文": topPapers(section),
  };
});
writeSheet(grouped, ["TPAMI主章节", "条目数", "P0数", "P1数", "预印本数", "章节核心论点", "建议比较维度", "代表论文"], groupedRows, {
  freezeColumns: 1,
  widths: {
    "TPAMI主章节": 240,
    "条目数": 70,
    "P0数": 70,
    "P1数": 70,
    "预印本数": 85,
    "章节核心论点": 520,
    "建议比较维度": 430,
    "代表论文": 620,
  },
});

const summary = workbook.worksheets.add("统计总览");
addTitle(summary, "统计总览", `共整理 ${organized.length} 条文献；建议先处理 P0/P1 与待核查条目。`);
const sectionCounts = countBy("TPAMI主章节").map(([k, v]) => [k, v]);
const priorityCounts = countBy("引用优先级").map(([k, v]) => [k, v]);
const contributionCounts = countBy("贡献类型").map(([k, v]) => [k, v]);
const evidenceCounts = countBy("证据强度").map(([k, v]) => [k, v]);
const yearCounts = countByYear().map(([k, v]) => [k, v]);

writeBlock(summary, 4, 1, [["TPAMI主章节", "数量"], ...sectionCounts], {
  borders: { preset: "inside", style: "thin", color: "#D1D5DB" },
});
writeBlock(summary, 4, 4, [["引用优先级", "数量"], ...priorityCounts], {
  borders: { preset: "inside", style: "thin", color: "#D1D5DB" },
});
writeBlock(summary, 4, 7, [["贡献类型", "数量"], ...contributionCounts], {
  borders: { preset: "inside", style: "thin", color: "#D1D5DB" },
});
writeBlock(summary, 20, 1, [["证据强度", "数量"], ...evidenceCounts], {
  borders: { preset: "inside", style: "thin", color: "#D1D5DB" },
});
writeBlock(summary, 20, 4, [["年份", "数量"], ...yearCounts], {
  borders: { preset: "inside", style: "thin", color: "#D1D5DB" },
});
summary.getRange("A4:B4").format.fill = "#1F4E79";
summary.getRange("D4:E4").format.fill = "#1F4E79";
summary.getRange("G4:H4").format.fill = "#1F4E79";
summary.getRange("A20:B20").format.fill = "#1F4E79";
summary.getRange("D20:E20").format.fill = "#1F4E79";
summary.getRange("A4:H20").format.font = { name: "Microsoft YaHei", size: 10 };
summary.getRange("A:A").format.columnWidthPx = 270;
summary.getRange("B:B").format.columnWidthPx = 70;
summary.getRange("D:D").format.columnWidthPx = 150;
summary.getRange("E:E").format.columnWidthPx = 70;
summary.getRange("G:G").format.columnWidthPx = 210;
summary.getRange("H:H").format.columnWidthPx = 70;
summary.freezePanes.freezeRows(3);
summary.charts.add("bar", {
  title: "按TPAMI主章节的文献分布",
  categories: sectionCounts.map((x) => x[0].replace(/^\d+_/, "")),
  series: [{ name: "数量", values: sectionCounts.map((x) => x[1]) }],
  hasLegend: false,
  barOptions: { direction: "bar", grouping: "clustered", gapWidth: 80 },
  dataLabels: { showValue: true, position: "outEnd" },
  from: { row: 3, col: 9 },
  extent: { widthPx: 780, heightPx: 360 },
});
summary.charts.add("ColumnClustered", {
  title: "按年份的文献数量",
  categories: yearCounts.map((x) => String(x[0])),
  series: [{ name: "数量", values: yearCounts.map((x) => x[1]) }],
  hasLegend: false,
  dataLabels: { showValue: true, position: "outEnd" },
  from: { row: 21, col: 7 },
  extent: { widthPx: 520, heightPx: 280 },
});

const outline = workbook.worksheets.add("TPAMI写作提纲");
const outlineRows = mainSections.map((s, i) => {
  const section = `${s.id}_${s.name}`;
  const must = organized
    .filter((r) => r["TPAMI主章节"] === section && (r["引用优先级"].startsWith("P0") || r["引用优先级"].startsWith("P1")))
    .sort((a, b) => a._priorityRank - b._priorityRank || (b["年份"] || 0) - (a["年份"] || 0))
    .slice(0, 10)
    .map((r) => r["论文名称"])
    .join("；");
  return {
    "建议顺序": i + 1,
    "章节": section,
    "段落目标": s.thesis,
    "必须覆盖论文": must || topPapers(section, 5),
    "对比维度": s.dimensions,
    "可写成的表/图": i === 5 ? "Mobile manipulation benchmark 对比表；指标从 SR 扩展到 TP/SRR/failure taxonomy" : i === 3 ? "真实数据集规模-模态-本体矩阵" : i === 4 ? "数据引擎路线图：sim/generative/real-to-sim/render/human-video" : "章节内代表论文对比表",
    "审稿人可能问": i === 1 ? "本文与已有 VLA survey/data survey 的差异是什么？" : i === 8 ? "为什么 success rate 不足以衡量 generalist robot？" : "这些工作如何共同支撑本文 taxonomy，而不是简单堆论文？",
  };
});
writeSheet(outline, ["建议顺序", "章节", "段落目标", "必须覆盖论文", "对比维度", "可写成的表/图", "审稿人可能问"], outlineRows, {
  freezeColumns: 2,
  widths: {
    "建议顺序": 70,
    "章节": 240,
    "段落目标": 520,
    "必须覆盖论文": 620,
    "对比维度": 420,
    "可写成的表/图": 360,
    "审稿人可能问": 360,
  },
});

const checks = workbook.worksheets.add("待核查");
const checkRows = organized
  .filter((r) => compact(r["核查标记"]))
  .map((r) => ({
    "论文名称": r["论文名称"],
    "Publication": r["Publication"],
    "年份": r["年份"],
    "引用优先级": r["引用优先级"],
    "证据强度": r["证据强度"],
    "核查标记": r["核查标记"],
    "建议动作": r["核查标记"].includes("缺分级")
      ? "补分级并决定是否进入主线"
      : r["核查标记"].includes("疑似重复")
        ? "合并重复记录或区分版本"
        : r["证据强度"].includes("arXiv")
          ? "投稿前查正式发表版本、代码和引用"
          : "核对来源可信度和最终题名",
    "链接": r["链接"],
  }));
writeSheet(checks, ["论文名称", "Publication", "年份", "引用优先级", "证据强度", "核查标记", "建议动作", "链接"], checkRows, {
  freezeColumns: 1,
  widths: {
    "论文名称": 420,
    "Publication": 150,
    "年份": 70,
    "引用优先级": 115,
    "证据强度": 150,
    "核查标记": 300,
    "建议动作": 320,
    "链接": 260,
  },
});

const raw = workbook.worksheets.add("原始数据");
const rawData = rows.map((row, idx) => {
  const out = {};
  for (const h of originalHeaders) out[h] = row[h] ?? "";
  if (!out["序号"]) out["序号"] = idx + 1;
  return out;
});
writeSheet(raw, originalHeaders, rawData, {
  freezeColumns: 2,
  widths: {
    "序号": 70,
    "论文名称": 360,
    "链接": 260,
    "Publication": 150,
    "Type": 240,
    "简要概括": 700,
    "数据量": 560,
    "评价指标": 560,
    "发布时间": 130,
    "分级": 100,
  },
});

// Add data validation on maintainable categorical columns in the organized table.
organizedSheet.getRange(`H2:H${organized.length + 1}`).dataValidation = {
  allowBlank: true,
  list: { inCellDropDown: true, source: Object.keys(priorityOrder) },
};
organizedSheet.getRange(`J2:J${organized.length + 1}`).dataValidation = {
  allowBlank: true,
  list: { inCellDropDown: true, source: mainSections.map((s) => `${s.id}_${s.name}`) },
};

await fs.mkdir(outputDir, { recursive: true });

// Compact verification before export.
const checksInspect = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
if (!checksInspect || /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A/.test(checksInspect.ndjson ?? "")) {
  throw new Error(`Formula/value error scan returned possible issues: ${checksInspect?.ndjson}`);
}

for (const sheetName of ["README_TPAMI", "分类字典", "整理总表", "按章节分组", "统计总览", "TPAMI写作提纲", "待核查", "原始数据"]) {
  await workbook.render({ sheetName, range: "A1:H20", scale: 1 });
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

console.log(JSON.stringify({
  outputPath,
  rows: organized.length,
  checkRows: checkRows.length,
  sheets: ["README_TPAMI", "分类字典", "整理总表", "按章节分组", "统计总览", "TPAMI写作提纲", "待核查", "原始数据"],
}, null, 2));
