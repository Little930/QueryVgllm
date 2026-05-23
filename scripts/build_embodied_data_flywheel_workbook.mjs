import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outputDir = path.join(root, "outputs", "tpami_review");
const sourcePath = path.join(outputDir, "source_rows.json");
const outputPath = path.join(outputDir, "TPAMI_具身数据飞轮综述整理.xlsx");

const rows = JSON.parse(await fs.readFile(sourcePath, "utf8"));

const rawHeaders = [
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

const flywheelHeaders = [
  "原序号",
  "论文名称",
  "链接",
  "Publication",
  "年份",
  "原Type",
  "原分级",
  "引用优先级",
  "数据主线",
  "数据来源",
  "飞轮环节",
  "训练作用",
  "VLM相关性",
  "VLA相关性",
  "主要对齐问题",
  "数据规模/信号",
  "模态",
  "机器人形态",
  "任务尺度",
  "是否长程",
  "是否失败/反馈数据",
  "是否部署闭环",
  "评测关注点",
  "适合写在哪一节",
  "一句话写法",
  "还欠什么",
  "数据量",
  "评价指标",
  "简要概括",
  "发布时间",
  "核查标记",
];

const routeDefs = [
  ["A_真机机器人数据", "高保真接触和真实分布，但采集贵、慢、异构；是 VLA grounding 的基准来源。"],
  ["B_仿真与生成式数据", "规模化和可控性强，但核心风险是物理真实性、视觉域差和 sim-to-real。"],
  ["C_人类视频/Ego数据", "最可扩展的人类行为来源，但动作不可直接执行，需要 hand/action/intent 对齐。"],
  ["D_Internet与VLM通用知识", "提供语义、视觉、常识和语言 grounding，是 VLA 泛化能力的上游。"],
  ["E_多源融合与跨本体", "把真机、仿真、人类视频、互联网数据放进统一训练配方，解决 representation/action/embodiment alignment。"],
  ["F_评测/推理/失败反馈", "数据飞轮闭环的质量控制层，覆盖 RoboVQA、长程任务、失败检测和部署反馈。"],
  ["G_硬件与采集接口", "UMI、ALOHA、GELLO 等降低真机数据采集门槛，是飞轮启动器。"],
  ["H_综述/定位/相邻领域", "用于Related Work和差异化定位，不作为主线证据。"],
];

const stageDefs = [
  ["Collect 采集", "真机遥操、人类视频、仿真轨迹、互联网图文视频、部署日志。"],
  ["Curate 组织清洗", "统一格式、去重、质量筛选、轨迹切分、失败/成功标签。"],
  ["Annotate 语义化", "语言指令、subtask、CoT、affordance、VQA、失败原因。"],
  ["Augment/Generate 扩增生成", "仿真生成、real-to-sim、real2render2real、任务生成、数据增强。"],
  ["Pre-train 预训练", "VLM/VLA 表征预训练、跨本体预训练、web knowledge 注入。"],
  ["Fine/Post-train 微调后训练", "真机微调、human+robot co-training、RL/偏好/失败修正。"],
  ["Evaluate 评测", "长程任务、开放词汇、RoboVQA、失败检测、泛化和鲁棒性。"],
  ["Deploy/Feedback 部署反馈", "机器人在真实环境中产生新成功/失败数据，回流再训练。"],
];

const priorityRank = { "P0 主线必引": 0, "P1 重点支撑": 1, "P2 背景比较": 2, "P3 可选/边缘": 3 };

function compact(v) {
  return String(v ?? "").replace(/\s+/g, " ").trim();
}

function lc(row) {
  return rawHeaders.map((h) => compact(row[h])).join(" ").toLowerCase();
}

function hasAny(text, words) {
  return words.some((w) => text.includes(w.toLowerCase()));
}

function yearOf(row) {
  const all = `${compact(row["发布时间"])} ${compact(row["Publication"])} ${compact(row["论文名称"])}`;
  let m = all.match(/(20\d{2})/);
  if (m) return Number(m[1]);
  m = all.match(/\b(ICCV|CVPR|ECCV|ICRA|IROS|RSS|CoRL|NeurIPS|ICLR|ICML|IJRR)\s*'?(\d{2})\b/i);
  return m ? 2000 + Number(m[2]) : "";
}

function routeOf(row) {
  const t = lc(row);
  const title = compact(row["论文名称"]).toLowerCase();
  const type = compact(row["Type"]).toLowerCase();
  const core = `${title} ${type}`;
  if (hasAny(core, ["survey", "综述", "review", "autonomous driving", "自驾", "3d generation", "world model"])) return "H_综述/定位/相邻领域";
  if (hasAny(core, ["aloha", "gello", "umi", "teleop", "遥操作", "采集硬件", "interface", "hardware", "bunny-visionpro", "pato"])) return "G_硬件与采集接口";
  if (hasAny(core, ["ego", "human video", "human-to-robot", "human to robot", "phanto", "being-h0", "egomimic", "egodex", "human manipulation", "人类 ego", "人类视频", "internet data"])) return "C_人类视频/Ego数据";
  if (hasAny(core, ["clip", "blip", "flamingo", "gpt", "llava", "vlm", "图文对齐", "visual instruction", "spatialvlm", "palm-e", "voxposer", "unipi", "navid"])) return "D_Internet与VLM通用知识";
  if (hasAny(core, ["simulation", "simulator", "仿真", "sim2real", "real-to-sim", "real2", "render", "synthetic", "合成", "robocasa", "maniskill", "genesis", "habitat", "sapien", "behavior", "omnigibson", "isaac", "mimicgen", "dexscale", "robogen", "mobilemanibench"])) return "B_仿真与生成式数据";
  if (hasAny(core, ["robovqa", "open eqa", "openeqa", "reasoning", "推理", "failure", "失败", "safe", "illusion", "benchmark", "bench"])) return "F_评测/推理/失败反馈";
  if (hasAny(core, ["open x-embodiment", "rt-x", "rt-2", "π0", "pi0", "pi 0", "gr-2", "robobrain", "gemini robotics", "omni", "wholebodyvla", "humanvla", "vla", "vlm4vla", "embodiedmidtrain", "cross-embodiment", "跨本体"])) return "E_多源融合与跨本体";
  if (hasAny(t, ["real-world", "真实", "真机", "in-the-wild", "dataset", "数据集", "droid", "bridge", "rh20t", "agibot", "brmdata", "airoa"])) return "A_真机机器人数据";
  return "A_真机机器人数据";
}

function dataSourceOf(row, route) {
  const t = lc(row);
  const src = [];
  if (hasAny(t, ["真实", "真机", "real-world", "real robot", "in-the-wild", "teleoperation", "遥操作", "spot", "fetch", "hsr", "agibot", "droid"])) src.push("Real robot");
  if (hasAny(t, ["simulation", "simulator", "仿真", "isaac", "habitat", "sapien", "omnigibson", "robocasa", "maniskill", "genesis"])) src.push("Simulation");
  if (hasAny(t, ["human video", "egocentric", "ego", "human-to-robot", "人类视频", "human data"])) src.push("Human/Ego video");
  if (hasAny(t, ["web", "internet", "图文", "vlm", "clip", "blip", "flamingo", "gpt", "llava"])) src.push("Internet/VLM");
  if (hasAny(t, ["synthetic", "合成", "render", "generative", "real-to-sim", "real2render", "zero robot"])) src.push("Synthetic/Generated");
  if (hasAny(t, ["benchmark", "robovqa", "eqa", "reasoning", "推理"])) src.push("Evaluation data");
  if (!src.length) src.push(route.replace(/^[A-H]_/, ""));
  return [...new Set(src)].join(" + ");
}

function stagesOf(row, route) {
  const t = lc(row);
  const s = [];
  if (route.startsWith("G") || hasAny(t, ["teleoperation", "遥操作", "采集", "hardware", "interface", "dataset"])) s.push("Collect 采集");
  if (hasAny(t, ["annotation", "标注", "subtask", "primitive", "language", "vqa", "reasoning", "cot"])) s.push("Annotate 语义化");
  if (route.startsWith("B") || hasAny(t, ["synthetic", "合成", "generate", "generative", "real-to-sim", "render", "sim2real", "mimicgen"])) s.push("Augment/Generate 扩增生成");
  if (route.startsWith("D") || route.startsWith("E") || hasAny(t, ["pre-train", "pretrain", "预训练", "foundation", "vla", "vlm", "robot brain"])) s.push("Pre-train 预训练");
  if (hasAny(t, ["fine-tune", "finetune", "post-training", "rl", "dpo", "sft", "policy optimization", "微调"])) s.push("Fine/Post-train 微调后训练");
  if (route.startsWith("F") || hasAny(t, ["benchmark", "evaluation", "评测", "success", "metric", "failure", "失败"])) s.push("Evaluate 评测");
  if (hasAny(t, ["deploy", "deployment", "fleet", "feedback", "on-the-fly", "correction", "失败数据", "failure data"])) s.push("Deploy/Feedback 部署反馈");
  if (!s.length) s.push(route.startsWith("H") ? "Curate 组织清洗" : "Collect 采集");
  return [...new Set(s)].join(" / ");
}

function trainingRole(row, route) {
  const t = lc(row);
  const roles = [];
  if (route.startsWith("D")) roles.push("VLM pretraining / semantic grounding");
  if (hasAny(t, ["vla", "vision-language-action", "robot brain", "robobrain", "rt-2", "π0", "pi0", "gr-2"])) roles.push("VLA pretraining");
  if (!route.startsWith("D") && hasAny(t, ["imitation", "behavior cloning", "teleoperation", "demonstration", "demonstrations", "模仿"])) roles.push("Policy learning / IL");
  if (hasAny(t, ["fine-tune", "finetune", "dpo", "sft", "rl", "post-training"])) roles.push("Fine/Post-training");
  if (route.startsWith("B") || hasAny(t, ["generate", "synthetic", "real-to-sim", "sim2real", "render"])) roles.push("Data generation / augmentation");
  if (route.startsWith("F") || hasAny(t, ["benchmark", "vqa", "eqa", "reasoning", "evaluation"])) roles.push("Evaluation / diagnosis");
  if (route.startsWith("C")) roles.push("Human-to-robot transfer");
  if (route.startsWith("E")) roles.push("Multi-source fusion / alignment");
  return [...new Set(roles)].join(" / ") || "Background support";
}

function vlmRelevance(row, route) {
  const t = lc(row);
  if (route.startsWith("D")) return "高：直接提供视觉-语言语义、常识或空间理解";
  if (route.startsWith("E") || hasAny(t, ["vla", "robot brain", "vlm", "language", "reasoning", "vqa", "open-vocabulary"])) return "中高：作为 VLA backbone、语言标注或推理监督";
  if (route.startsWith("F")) return "中：用于评估语义理解、推理和任务解释";
  return "低到中：主要是动作/交互数据，但可通过语言标注接入 VLM";
}

function vlaRelevance(row, route) {
  if (["A", "B", "C", "E", "F", "G"].some((p) => route.startsWith(p))) return "高：直接影响 VLA 的动作 grounding、泛化或评测";
  if (route.startsWith("D")) return "中：提供上游 VLM 知识，需通过动作数据落地";
  return "背景：用于定位和对比";
}

function alignmentIssue(row, route) {
  const t = lc(row);
  const issues = [];
  if (hasAny(t, ["cross-embodiment", "跨本体", "x-embodiment", "different robot", "heterogeneous"])) issues.push("Embodiment alignment");
  if (hasAny(t, ["human", "ego", "hand", "human-to-robot"])) issues.push("Human-robot action alignment");
  if (hasAny(t, ["simulation", "sim2real", "real-to-sim", "render", "synthetic"])) issues.push("Visual/physics domain gap");
  if (hasAny(t, ["language", "instruction", "vlm", "vqa", "reasoning", "open-vocabulary"])) issues.push("Language/semantic grounding");
  if (hasAny(t, ["long-horizon", "长程", "subtask", "task progress"])) issues.push("Temporal/subtask alignment");
  if (hasAny(t, ["failure", "失败", "re-plan", "correction"])) issues.push("Failure/recovery supervision");
  if (!issues.length) {
    if (route.startsWith("A")) issues.push("Action space and embodiment heterogeneity");
    else if (route.startsWith("B")) issues.push("Physical realism and sim-to-real");
    else if (route.startsWith("C")) issues.push("Human action to robot action");
    else if (route.startsWith("D")) issues.push("VLM semantics to executable actions");
    else issues.push("Taxonomy/metric alignment");
  }
  return [...new Set(issues)].join(" / ");
}

function scaleSignal(row) {
  const data = compact(row["数据量"]);
  const summary = compact(row["简要概括"]);
  const m = `${data} ${summary}`.match(/((?:\d+[,.]?\d*|\d+)\s*(?:K|M|B|k|m|b|万|亿|hours?|小时|episodes?|trajectories?|tasks?|scenes?|objects?|demonstrations?)[^。；;]*)/);
  if (data) return data.slice(0, 260);
  return m ? m[1].slice(0, 260) : "待补：规模、任务数、小时数或轨迹数";
}

function modalities(row) {
  const t = lc(row);
  const mods = [];
  if (hasAny(t, ["rgb", "image", "视觉", "camera", "video"])) mods.push("RGB/video");
  if (hasAny(t, ["depth", "point cloud", "3d", "点云"])) mods.push("Depth/3D");
  if (hasAny(t, ["language", "instruction", "text", "语言", "vqa", "cot"])) mods.push("Language");
  if (hasAny(t, ["action", "trajectory", "joint", "end-effector", "pose", "动作", "轨迹"])) mods.push("Action/trajectory");
  if (hasAny(t, ["force", "torque", "tactile", "触觉", "力"])) mods.push("Force/tactile");
  if (hasAny(t, ["state", "proprio", "关节", "robot states"])) mods.push("Robot state");
  if (!mods.length) mods.push("待补");
  return [...new Set(mods)].join(" + ");
}

function embodiment(row) {
  const t = lc(row);
  const e = [];
  if (hasAny(t, ["mobile", "移动", "base", "底盘", "hsr", "fetch", "spot"])) e.push("移动底盘+机械臂");
  if (hasAny(t, ["single arm", "manipulator", "机械臂", "robot arm"])) e.push("机械臂");
  if (hasAny(t, ["bimanual", "双臂", "aloha"])) e.push("双臂");
  if (hasAny(t, ["dex", "灵巧", "hand", "grasp", "gripper"])) e.push("手/夹爪");
  if (hasAny(t, ["humanoid", "人形", "whole-body", "loco"])) e.push("人形/全身");
  if (hasAny(t, ["navigation", "vln"])) e.push("导航智能体");
  if (hasAny(t, ["cross-embodiment", "x-embodiment", "跨本体"])) e.push("跨本体");
  return [...new Set(e)].join(" + ") || "通用/未限定";
}

function taskScale(row) {
  const t = lc(row);
  if (hasAny(t, ["long-horizon", "long horizon", "长程", "everyday", "household", "日常", "家庭"])) return "长程家庭/开放任务";
  if (hasAny(t, ["navigation", "vln", "objectnav", "导航"])) return "导航/移动";
  if (hasAny(t, ["grasp", "in-hand", "handover", "抓取"])) return "抓取/接触技能";
  if (hasAny(t, ["vqa", "eqa", "reasoning", "question"])) return "问答/推理";
  return "短程操作/通用能力";
}

function yesNoLong(row) {
  return hasAny(lc(row), ["long-horizon", "long horizon", "长程", "everyday", "household", "日常"]) ? "是" : "未明确/否";
}

function failureFeedback(row) {
  const t = lc(row);
  if (hasAny(t, ["failure", "失败", "error", "re-plan", "replan", "correction", "feedback", "on-the-fly"])) return "是：含失败、纠错、re-plan 或反馈信号";
  return "未明确";
}

function deployLoop(row) {
  const t = lc(row);
  if (hasAny(t, ["deploy", "deployment", "fleet", "in-the-wild", "real-world", "on-the-fly", "feedback"])) return "有部署/真实环境线索";
  if (hasAny(t, ["benchmark", "simulation", "仿真"])) return "主要是离线/仿真闭环";
  return "未明确";
}

function evalFocus(row) {
  const t = lc(row);
  const f = [];
  if (hasAny(t, ["success rate", "sr", "成功率", "success"])) f.push("SR");
  if (hasAny(t, ["task progress", "tp", "progress"])) f.push("Task progress");
  if (hasAny(t, ["failure", "失败", "re-plan", "srr"])) f.push("Failure/re-plan");
  if (hasAny(t, ["generalization", "unseen", "泛化"])) f.push("Generalization");
  if (hasAny(t, ["sim2real", "real-to-sim"])) f.push("Sim2Real");
  if (hasAny(t, ["spatial", "空间"])) f.push("Spatial reasoning");
  if (hasAny(t, ["vqa", "eqa", "reasoning", "推理"])) f.push("Embodied reasoning");
  if (hasAny(t, ["scaling", "scale"])) f.push("Scaling");
  return f.join(" / ") || "规模、覆盖、泛化、任务成功";
}

function priority(row, route) {
  const level = compact(row["分级"]);
  const title = compact(row["论文名称"]).toLowerCase();
  if (["必引", "核心前沿", "核心补充"].includes(level)) return "P0 主线必引";
  if (hasAny(title, ["rt-2", "open x-embodiment", "droid", "mobile aloha", "home-robot", "ok-robot", "gr-2", "π0.5", "mobilemanibench", "emmoe", "lambda", "airoa"])) return "P0 主线必引";
  if (["重要", "Insight"].includes(level)) return "P1 重点支撑";
  if (["可选", "marginal"].includes(level)) return "P3 可选/边缘";
  if (route.startsWith("H")) return "P2 背景比较";
  return "P2 背景比较";
}

function sectionPlacement(route, stage) {
  if (route.startsWith("A")) return "III. Real-robot data: fidelity, heterogeneity, and collection bottlenecks";
  if (route.startsWith("B")) return "IV. Simulation/generative data: scalable but physically grounded?";
  if (route.startsWith("C")) return "V. Human video and ego data: scalable human experience";
  if (route.startsWith("D")) return "II. VLM and internet knowledge as upstream embodied data";
  if (route.startsWith("E")) return "VI. Multi-source fusion and cross-embodiment VLA training";
  if (route.startsWith("F")) return "VII. Evaluation, reasoning, failure data, and deployment feedback";
  if (route.startsWith("G")) return "III-A. Hardware-assisted data collection";
  return "I/Related Work. Positioning against existing surveys";
}

function oneLine(row, route, stage, role) {
  const title = compact(row["论文名称"]);
  if (route.startsWith("A")) return `${title} 可作为真机高保真数据证据，重点写数据规模、机器人异构性、模态和采集成本。`;
  if (route.startsWith("B")) return `${title} 可用于说明仿真/生成式数据如何扩展任务和场景，以及 sim-to-real 风险。`;
  if (route.startsWith("C")) return `${title} 支撑“人类经验可扩展但需动作对齐”的论点。`;
  if (route.startsWith("D")) return `${title} 用来说明 VLM/互联网知识如何给 VLA 提供语义、空间和常识先验。`;
  if (route.startsWith("E")) return `${title} 放在多源融合/跨本体训练配方中，强调 ${role || "alignment"}。`;
  if (route.startsWith("F")) return `${title} 用于飞轮的评测和诊断层，说明模型能力、失败和推理如何被量化。`;
  if (route.startsWith("G")) return `${title} 说明硬件/接口如何降低真机数据采集门槛，是数据飞轮冷启动工具。`;
  return `${title} 用于 Related Work 或方法论定位，帮助说明本文与已有综述的差异。`;
}

function missing(row, route) {
  const m = [];
  if (!compact(row["数据量"])) m.push("补数据规模");
  if (!compact(row["评价指标"])) m.push("补评测指标");
  if (!compact(row["链接"])) m.push("补链接");
  if (!compact(row["分级"])) m.push("补引用优先级");
  if (route.startsWith("A") && !hasAny(lc(row), ["failure", "失败"])) m.push("查是否有失败/部署反馈");
  if (route.startsWith("C")) m.push("查动作表示/人到机器人对齐方式");
  if (route.startsWith("B")) m.push("查物理真实性与sim-to-real证据");
  if (route.startsWith("D")) m.push("补VLM数据如何进入VLA训练");
  return [...new Set(m)].join("；") || "基本可用";
}

function checkFlag(row) {
  const flags = [];
  const pub = compact(row["Publication"]);
  if (!compact(row["链接"])) flags.push("缺链接");
  if (!compact(row["分级"])) flags.push("缺分级");
  if (!yearOf(row)) flags.push("缺年份");
  if (/arxiv/i.test(pub)) flags.push("arXiv版本待核");
  if (/blog|system card|technical report|researchgate|preprint|在投/i.test(pub)) flags.push("非正式发表/待核");
  return flags.join("；");
}

function routeNarrative(route) {
  return routeDefs.find(([r]) => r === route)?.[1] ?? "";
}

const flywheelRows = rows.map((row, idx) => {
  const route = routeOf(row);
  const stage = stagesOf(row, route);
  const role = trainingRole(row, route);
  return {
    "原序号": row["序号"] || idx + 1,
    "论文名称": compact(row["论文名称"]),
    "链接": compact(row["链接"]),
    "Publication": compact(row["Publication"]),
    "年份": yearOf(row),
    "原Type": compact(row["Type"]),
    "原分级": compact(row["分级"]),
    "引用优先级": priority(row, route),
    "数据主线": route,
    "数据来源": dataSourceOf(row, route),
    "飞轮环节": stage,
    "训练作用": role,
    "VLM相关性": vlmRelevance(row, route),
    "VLA相关性": vlaRelevance(row, route),
    "主要对齐问题": alignmentIssue(row, route),
    "数据规模/信号": scaleSignal(row),
    "模态": modalities(row),
    "机器人形态": embodiment(row),
    "任务尺度": taskScale(row),
    "是否长程": yesNoLong(row),
    "是否失败/反馈数据": failureFeedback(row),
    "是否部署闭环": deployLoop(row),
    "评测关注点": evalFocus(row),
    "适合写在哪一节": sectionPlacement(route, stage),
    "一句话写法": oneLine(row, route, stage, role),
    "还欠什么": missing(row, route),
    "数据量": compact(row["数据量"]),
    "评价指标": compact(row["评价指标"]),
    "简要概括": compact(row["简要概括"]),
    "发布时间": compact(row["发布时间"]),
    "核查标记": checkFlag(row),
    _rank: priorityRank[priority(row, route)] ?? 9,
  };
}).sort((a, b) => a["数据主线"].localeCompare(b["数据主线"], "zh-Hans-CN") || a._rank - b._rank || (b["年份"] || 0) - (a["年份"] || 0));

function col(n) {
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function writeSheet(sheet, headers, data, widths = {}, freezeCols = 1) {
  const values = [headers, ...data.map((r) => headers.map((h) => r[h] ?? ""))];
  sheet.getRange(`A1:${col(headers.length)}${values.length}`).values = values;
  const used = sheet.getRange(`A1:${col(headers.length)}${values.length}`);
  used.format.font = { name: "Microsoft YaHei", size: 10, color: "#111827" };
  used.format.verticalAlignment = "top";
  used.format.wrapText = true;
  used.format.borders = { preset: "inside", style: "thin", color: "#E5E7EB" };
  const header = sheet.getRange(`A1:${col(headers.length)}1`);
  header.format.fill = "#204060";
  header.format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
  header.format.horizontalAlignment = "center";
  sheet.freezePanes.freezeRows(1);
  if (freezeCols) sheet.freezePanes.freezeColumns(freezeCols);
  headers.forEach((h, i) => {
    sheet.getRange(`${col(i + 1)}1:${col(i + 1)}${values.length}`).format.columnWidthPx = widths[h] ?? 140;
  });
}

function writeBlock(sheet, row, colIndex, values, style = {}) {
  const r = sheet.getRange(`${col(colIndex)}${row}:${col(colIndex + values[0].length - 1)}${row + values.length - 1}`);
  r.values = values;
  r.format = {
    font: { name: "Microsoft YaHei", size: 10, color: "#111827" },
    verticalAlignment: "top",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#D1D5DB" },
    ...style,
  };
  return r;
}

function title(sheet, text, subtitle) {
  writeBlock(sheet, 1, 1, [[text]], {
    fill: "#204060",
    font: { name: "Microsoft YaHei", size: 16, color: "#FFFFFF", bold: true },
  });
  sheet.getRange("A1:H1").merge();
  writeBlock(sheet, 2, 1, [[subtitle]], {
    fill: "#EAF3F8",
    font: { name: "Microsoft YaHei", size: 10, color: "#1F2937" },
  });
  sheet.getRange("A2:H2").merge();
}

function countBy(key) {
  const m = new Map();
  for (const r of flywheelRows) m.set(r[key], (m.get(r[key]) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0]), "zh-Hans-CN"));
}

function stageCounts() {
  const m = new Map();
  for (const r of flywheelRows) {
    for (const s of compact(r["飞轮环节"]).split(" / ")) {
      m.set(s, (m.get(s) ?? 0) + 1);
    }
  }
  return [...m.entries()].sort((a, b) => b[1] - a[1]);
}

function topPapers(route, n = 8) {
  return flywheelRows
    .filter((r) => r["数据主线"] === route)
    .sort((a, b) => a._rank - b._rank || (b["年份"] || 0) - (a["年份"] || 0))
    .slice(0, n)
    .map((r) => `${r["论文名称"]} (${r["年份"] || "n.d."}, ${r["引用优先级"].slice(0, 2)})`)
    .join("；");
}

const workbook = Workbook.create();

const readme = workbook.worksheets.getOrAdd("README_数据飞轮", { renameFirstIfOnlyNewSpreadsheet: true });
title(readme, "具身数据飞轮综述整理版", "按“数据从哪里来、怎么进入训练、如何评估和部署反馈”重构，避免写成普通 VLA 数据集列表。");
writeBlock(readme, 4, 1, [
  ["核心叙事", "真机数据高保真但难以 scale；仿真、人类视频、Internet/VLM 知识和生成式数据补足规模；多源融合训练 VLA；评测/部署产生新数据，形成数据飞轮。"],
  ["和已有综述区别", "不是只列 datasets / benchmarks / data engines，而是分析这些数据如何在 Collect-Curate-Annotate-Augment-Train-Evaluate-Deploy 之间流动。"],
  ["当前最该补", "VLM 通用知识如何进入 VLA；RoboVQA/RoboBrain/embodied reasoning；动态数据集、失败数据、部署反馈；human video 与真机数据共训练。"],
  ["使用建议", "先看「方向汇总」定框架，再用「飞轮证据表」筛 P0/P1，最后看「差异化定位」写 introduction 和 related work。"],
], { borders: { preset: "inside", style: "thin", color: "#D1D5DB" } });
readme.getRange("A:A").format.columnWidthPx = 160;
readme.getRange("B:B").format.columnWidthPx = 980;

const framework = workbook.worksheets.add("飞轮框架");
title(framework, "Embodied Data Flywheel 框架", "这页用于把论文从静态资源表转成闭环机制。");
writeBlock(framework, 4, 1, [
  ["飞轮环节", "核心问题", "典型数据/方法", "写作时要问"],
  ["Collect 采集", "数据从哪里来？", "真机遥操、硬件采集、人类视频、仿真轨迹、互联网图文视频、部署日志", "成本、规模、模态、机器人异构性如何？"],
  ["Curate 组织清洗", "如何把乱数据变成可训练数据？", "统一格式、轨迹切分、质量筛选、去重、失败/成功标签", "是否支持跨数据集比较和复用？"],
  ["Annotate 语义化", "如何把行为变成语言和推理监督？", "指令、subtask、primitive action、VQA、CoT、失败原因", "标注粒度是否足以训练长程 VLA？"],
  ["Augment/Generate 扩增生成", "如何低成本扩大覆盖？", "仿真、real-to-sim、real2render2real、任务生成、synthetic action data", "物理真实性和 sim-to-real 如何验证？"],
  ["Pre-train 预训练", "如何形成可迁移表征？", "VLM/VLA 预训练、跨本体数据混合、Internet knowledge", "VLM 语义如何接到 action grounding？"],
  ["Fine/Post-train 微调后训练", "如何获得具体机器人能力？", "真机微调、human+robot co-training、RL、DPO/SFT、失败修正", "是否证明了新数据源带来的增益？"],
  ["Evaluate 评测", "如何判断模型真的变强？", "RoboVQA、OpenEQA、长程任务、task progress、failure/re-plan", "是否超越简单 success rate？"],
  ["Deploy/Feedback 部署反馈", "如何形成持续变强闭环？", "真实部署、fleet data、失败轨迹回收、主动补采", "是否有在线/持续学习证据？"],
], { borders: { preset: "inside", style: "thin", color: "#D1D5DB" } });
framework.getRange("A4:D4").format.fill = "#204060";
framework.getRange("A4:D4").format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
framework.getRange("A:A").format.columnWidthPx = 180;
framework.getRange("B:B").format.columnWidthPx = 260;
framework.getRange("C:C").format.columnWidthPx = 420;
framework.getRange("D:D").format.columnWidthPx = 420;
framework.freezePanes.freezeRows(4);

const evidence = workbook.worksheets.add("飞轮证据表");
writeSheet(evidence, flywheelHeaders, flywheelRows, {
  "原序号": 70,
  "论文名称": 360,
  "链接": 260,
  "Publication": 150,
  "年份": 70,
  "原Type": 210,
  "原分级": 95,
  "引用优先级": 110,
  "数据主线": 180,
  "数据来源": 180,
  "飞轮环节": 220,
  "训练作用": 260,
  "VLM相关性": 300,
  "VLA相关性": 260,
  "主要对齐问题": 300,
  "数据规模/信号": 420,
  "模态": 210,
  "机器人形态": 190,
  "任务尺度": 160,
  "是否长程": 110,
  "是否失败/反馈数据": 220,
  "是否部署闭环": 180,
  "评测关注点": 240,
  "适合写在哪一节": 380,
  "一句话写法": 520,
  "还欠什么": 280,
  "数据量": 520,
  "评价指标": 520,
  "简要概括": 700,
  "发布时间": 130,
  "核查标记": 230,
}, 2);
evidence.getRange(`H2:H${flywheelRows.length + 1}`).conditionalFormats.add("containsText", {
  text: "P0",
  format: { fill: "#D1FAE5", font: { color: "#065F46", bold: true } },
});
evidence.getRange(`I2:I${flywheelRows.length + 1}`).conditionalFormats.add("containsText", {
  text: "D_Internet",
  format: { fill: "#DBEAFE", font: { color: "#1D4ED8", bold: true } },
});
evidence.getRange(`U2:V${flywheelRows.length + 1}`).conditionalFormats.add("containsText", {
  text: "是",
  format: { fill: "#FEF3C7", font: { color: "#92400E", bold: true } },
});

const summary = workbook.worksheets.add("方向汇总");
title(summary, "方向汇总", "用于快速判断每条主线现在有多少文献、主张是什么、还欠什么。");
const routeSummary = routeDefs.map(([route, claim]) => {
  const subset = flywheelRows.filter((r) => r["数据主线"] === route);
  return {
    "数据主线": route,
    "数量": subset.length,
    "P0/P1数量": subset.filter((r) => r["引用优先级"].startsWith("P0") || r["引用优先级"].startsWith("P1")).length,
    "主张": claim,
    "代表论文": topPapers(route),
    "目前短板": route.startsWith("D")
      ? "需要补更多 VLM data / VQA / spatial reasoning 如何帮助 VLA 的论文"
      : route.startsWith("F")
        ? "需要补部署反馈、失败数据、动态评测和 RoboVQA/RoboBrain 系列"
        : route.startsWith("C")
          ? "需要补 human video 到 action supervision 的最新工作"
          : "需要补数据规模、指标、是否有开源数据和正式发表版本",
  };
});
writeSheet(summary, ["数据主线", "数量", "P0/P1数量", "主张", "代表论文", "目前短板"], routeSummary, {
  "数据主线": 190,
  "数量": 70,
  "P0/P1数量": 90,
  "主张": 420,
  "代表论文": 680,
  "目前短板": 420,
}, 1);

const stats = workbook.worksheets.add("统计总览");
title(stats, "统计总览", `共 ${flywheelRows.length} 条；新版以数据飞轮而不是普通章节作为主视角。`);
const routeCounts = countBy("数据主线");
const priorityCounts = countBy("引用优先级");
const stageCountRows = stageCounts();
writeBlock(stats, 4, 1, [["数据主线", "数量"], ...routeCounts], { borders: { preset: "inside", style: "thin", color: "#D1D5DB" } });
writeBlock(stats, 4, 4, [["飞轮环节", "数量"], ...stageCountRows], { borders: { preset: "inside", style: "thin", color: "#D1D5DB" } });
writeBlock(stats, 4, 7, [["引用优先级", "数量"], ...priorityCounts], { borders: { preset: "inside", style: "thin", color: "#D1D5DB" } });
stats.getRange("A4:B4").format.fill = "#204060";
stats.getRange("D4:E4").format.fill = "#204060";
stats.getRange("G4:H4").format.fill = "#204060";
stats.getRange("A4:H4").format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
stats.getRange("A:A").format.columnWidthPx = 190;
stats.getRange("D:D").format.columnWidthPx = 220;
stats.getRange("G:G").format.columnWidthPx = 130;
stats.charts.add("bar", {
  title: "按数据主线分布",
  categories: routeCounts.map((x) => x[0]),
  series: [{ name: "数量", values: routeCounts.map((x) => x[1]) }],
  hasLegend: false,
  barOptions: { direction: "bar", grouping: "clustered", gapWidth: 70 },
  dataLabels: { showValue: true, position: "outEnd" },
  from: { row: 3, col: 9 },
  extent: { widthPx: 760, heightPx: 350 },
});
stats.charts.add("ColumnClustered", {
  title: "按飞轮环节分布",
  categories: stageCountRows.map((x) => x[0].replace(/ .*/, "")),
  series: [{ name: "数量", values: stageCountRows.map((x) => x[1]) }],
  hasLegend: false,
  dataLabels: { showValue: true, position: "outEnd" },
  from: { row: 20, col: 4 },
  extent: { widthPx: 620, heightPx: 280 },
});

const differentiation = workbook.worksheets.add("差异化定位");
title(differentiation, "与已有综述的差异化定位", "这页可以直接转成 introduction/related work 的 positioning。");
writeBlock(differentiation, 4, 1, [
  ["对比对象", "它主要讲什么", "我们的差异化切入", "写作提醒"],
  ["Vision-Language-Action in Robotics: A Survey of Datasets, Benchmarks, and Data Engines", "按 datasets / benchmarks / data engines 三类组织，强调数据基础设施和评测协议。", "我们不只列资源，而是追踪数据如何在采集、标注、生成、训练、评测、部署反馈之间形成闭环。", "避免重复三分法；把该文作为相关综述承认，并说明本文关注 data flywheel mechanism。"],
  ["Large VLM-based VLA Models for Robotic Manipulation: A Survey", "偏方法和模型 taxonomy，关注 large VLM-based VLA 的架构、范式和 manipulation。", "我们以数据为中心，讨论 VLM 通用知识、真机/仿真/human video 如何共同支撑 VLA，而不是主讲模型结构。", "可以引用其方法 taxonomy，但正文重点放在数据来源、对齐、融合和反馈。"],
  ["PI / Physical Intelligence 系列", "强调大规模多样化机器人数据、human-to-robot transfer 和数据使用配方。", "作为强案例，不作为全文唯一主线；用于说明 real robot + human data + post-training 的飞轮实践。", "师兄提醒不要以 PI 为唯一叙事，应作为 case study。"],
  ["Data Pyramid / Data Flywheel talk", "提出 robotic foundation model 的数据金字塔和数据飞轮原则。", "可作为综述框架灵感，但需要用论文证据把每个环节落地。", "不要只复述概念，要把表里的论文映射到每个环节。"],
], { borders: { preset: "inside", style: "thin", color: "#D1D5DB" } });
differentiation.getRange("A4:D4").format.fill = "#204060";
differentiation.getRange("A4:D4").format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
differentiation.getRange("A:A").format.columnWidthPx = 310;
differentiation.getRange("B:D").format.columnWidthPx = 420;

const writing = workbook.worksheets.add("写作骨架");
title(writing, "建议初稿骨架", "先按这个结构写一个粗版，再补论文和图表。");
writeBlock(writing, 4, 1, [
  ["章节", "核心问题", "建议用的论文主线", "建议图表"],
  ["1 Introduction", "为什么具身数据比单纯模型架构更关键？为什么需要 data flywheel？", "差异化定位 + P0/P1", "Figure 1: Embodied Data Flywheel"],
  ["2 VLM/Internet Knowledge as Upstream Embodied Data", "VLM 的通用知识如何帮助 VLA，而它缺什么？", "D_Internet与VLM通用知识", "表：VLM能力 -> VLA训练作用"],
  ["3 Real-Robot Data and Hardware-Assisted Collection", "真机数据为什么最可靠但难 scale？异构性怎么处理？", "A_真机机器人数据 + G_硬件与采集接口", "表：真机数据集规模/机器人/模态/任务"],
  ["4 Simulation and Generative Data Engines", "仿真/生成如何补规模？real2sim/sim2real/real2sim2real 怎么分？", "B_仿真与生成式数据", "图：仿真数据路线树"],
  ["5 Human/Ego Video for Robot Learning", "人类视频如何变成可执行监督？何时有用？", "C_人类视频/Ego数据", "表：第一视角/第三视角/hand tracking/latent action"],
  ["6 Multi-Source Fusion for VLA Training", "如何把真机、仿真、人类视频、VLM数据合到一个训练配方？", "E_多源融合与跨本体", "图：pretraining-finetuning-posttraining 数据混合"],
  ["7 Evaluation, Reasoning, Failure, and Feedback", "飞轮怎么知道自己变强了？失败数据如何回流？", "F_评测/推理/失败反馈", "表：SR之外的指标"],
  ["8 Open Challenges", "对齐、质量、可扩展、部署安全、数据治理。", "全表", "总结表：挑战 -> 现有证据 -> 空白"],
], { borders: { preset: "inside", style: "thin", color: "#D1D5DB" } });
writing.getRange("A4:D4").format.fill = "#204060";
writing.getRange("A4:D4").format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
writing.getRange("A:A").format.columnWidthPx = 270;
writing.getRange("B:D").format.columnWidthPx = 450;

const gaps = workbook.worksheets.add("补文献方向");
title(gaps, "接下来最该补的论文方向", "按师兄反馈：多加，相关先放表里，不相关后面筛。");
writeBlock(gaps, 4, 1, [
  ["优先级", "方向", "为什么要补", "搜索关键词/例子"],
  ["高", "VLM通用知识与VLA训练", "避免只写 manipulation data；说明 VLM 语义、空间、常识如何进入 VLA。", "VLM for robotics, embodied VQA, spatial VLM, VLM to VLA, mid-training"],
  ["高", "RoboVQA / RoboBrain / embodied reasoning", "和普通数据集综述拉开差异；覆盖推理、问答、失败诊断。", "RoboVQA, RoboBrain, OpenEQA, embodied reasoning benchmark, failure detection VLA"],
  ["高", "Human/Ego video 到 robot policy", "师兄特别提到 PI 和 human-to-robot；这是数据飞轮的重要 scale 路线。", "human-to-robot transfer, ego video imitation learning, latent action, hand tracking VLA"],
  ["高", "部署反馈和失败数据", "真正的数据飞轮必须有 deploy -> feedback -> retrain。当前表里这块明显少。", "robot data flywheel, fleet learning robotics, failure data, on-the-fly correction"],
  ["中", "Navigation + Manipulation / Mobile humanoid data", "师兄提到不仅是机械臂，还要既能走又能抓取。", "mobile manipulation dataset, humanoid manipulation dataset, loco-manipulation VLA"],
  ["中", "Real2Sim / Sim2Real / Real2Sim2Real细分", "仿真路线需要分清，不能只笼统写 simulation。", "real-to-sim robotics, sim-to-real VLA, real2render2real, generative simulation"],
  ["中", "动态/长程数据集和评测", "长程任务、动态环境、task progress 是评测短板。", "long-horizon robotic dataset, dynamic embodied benchmark, task progress metric"],
  ["中", "数据治理和质量筛选", "TPAMI 综述需要讲数据质量、覆盖、去重、标注成本，不只是规模。", "robot data curation, dataset quality, data filtering robotics"],
], { borders: { preset: "inside", style: "thin", color: "#D1D5DB" } });
gaps.getRange("A4:D4").format.fill = "#204060";
gaps.getRange("A4:D4").format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
gaps.getRange("A:A").format.columnWidthPx = 90;
gaps.getRange("B:B").format.columnWidthPx = 280;
gaps.getRange("C:C").format.columnWidthPx = 520;
gaps.getRange("D:D").format.columnWidthPx = 520;

const raw = workbook.worksheets.add("原始数据");
const rawRows = rows.map((r, i) => {
  const out = {};
  for (const h of rawHeaders) out[h] = r[h] ?? "";
  if (!out["序号"]) out["序号"] = i + 1;
  return out;
});
writeSheet(raw, rawHeaders, rawRows, {
  "序号": 70,
  "论文名称": 360,
  "链接": 260,
  "Publication": 150,
  "Type": 230,
  "简要概括": 700,
  "数据量": 560,
  "评价指标": 560,
  "发布时间": 130,
  "分级": 100,
}, 2);

// Maintainable dropdowns on the main evidence sheet.
evidence.getRange(`I2:I${flywheelRows.length + 1}`).dataValidation = {
  allowBlank: true,
  list: { inCellDropDown: true, source: routeDefs.map(([r]) => r) },
};
evidence.getRange(`H2:H${flywheelRows.length + 1}`).dataValidation = {
  allowBlank: true,
  list: { inCellDropDown: true, source: Object.keys(priorityRank) },
};

await fs.mkdir(outputDir, { recursive: true });
const err = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
if (/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A/.test(err.ndjson ?? "")) {
  throw new Error(`Workbook error scan found issues: ${err.ndjson}`);
}

for (const sheetName of [
  "README_数据飞轮",
  "飞轮框架",
  "飞轮证据表",
  "方向汇总",
  "统计总览",
  "差异化定位",
  "写作骨架",
  "补文献方向",
  "原始数据",
]) {
  await workbook.render({ sheetName, range: "A1:H20", scale: 1 });
}

const out = await SpreadsheetFile.exportXlsx(workbook);
await out.save(outputPath);

console.log(JSON.stringify({
  outputPath,
  rows: flywheelRows.length,
  routeCounts,
  stageCountRows,
}, null, 2));
