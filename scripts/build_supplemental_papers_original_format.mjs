import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outputDir = path.join(root, "outputs", "tpami_review");
const outputPath = path.join(outputDir, "补充文献_部署反馈失败数据质量安全_原表格式.xlsx");

const headers = ["序号", "论文名称", "链接", "Publication", "Type", "简要概括", "数据量", "评价指标", "发布时间", "分级"];

const rows = [
  {
    "序号": 114,
    "论文名称": "Robot-Powered Data Flywheels: Deploying Robots in the Wild for Continual Data Collection and Foundation Model Adaptation",
    "链接": "https://arxiv.org/abs/2511.19647",
    "Publication": "arXiv 2025",
    "Type": "数据飞轮闭环 / 真实部署数据 / Foundation Model Adaptation / Robot-generated Data",
    "简要概括": "本文非常适合放在综述的数据飞轮主线中。它明确提出 Robot-Powered Data Flywheel，把机器人从 foundation model 的使用者转变成真实世界数据生成者：机器人在真实环境中执行有用任务，同时采集现有互联网预训练数据中缺失的复杂、遮挡、多语言、长尾视觉数据，再用这些部署数据反过来微调 foundation model。论文用 Scanford 移动操作机器人在 Stanford East Asia Library 的真实部署作为例子，机器人自主扫描书架，用 VLM 识别书籍，并结合图书馆 catalog 自动标注图像，从而形成“部署-采集-自动标注-微调-提升模型”的闭环。对综述来说，这篇可以作为 data flywheel 概念的核心证据：它不是离线数据集，而是真正展示机器人如何在真实部署中持续产生数据并改善模型。",
    "数据量": "Scanford 在 East Asia Library 真实部署 2 周，采集来自 2,103 个书架的真实图像数据。任务是移动机器人扫描书架、识别书籍、用图书馆 catalog 自动产生标签，不需要人工逐张标注。数据分布包括遮挡书脊、多语言文本、真实光照和书架摆放等互联网图文数据中欠覆盖的长尾情况。论文还报告该系统节省约 18.7 小时人工扫描/录入时间。",
    "评价指标": "主要评估 foundation model adaptation 的收益：书籍识别准确率从 32.0% 提升到 71.8%；domain-adjacent multilingual OCR 上，英文 OCR 从 24.8% 提升到 46.6%，中文 OCR 从 30.8% 提升到 38.0%。评价价值在于证明机器人部署数据不仅提升当前场景，还能改善相邻域能力。",
    "发布时间": "2025/11/24",
    "分级": "核心前沿",
  },
  {
    "序号": 115,
    "论文名称": "Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment",
    "链接": "https://arxiv.org/abs/2211.08416",
    "Publication": "RSS 2023 / IJRR 2025",
    "Type": "部署中学习 / Human-in-the-loop Deployment / Intervention Data / 数据重加权",
    "简要概括": "本文提出 Sirius，是部署反馈方向非常重要的一篇。它的核心问题是：机器人策略在真实任务中不可能一开始就完全可靠，但可以通过人机分工安全部署，并把人类接管/干预数据转化为后续训练信号。Sirius 框架中，机器人在自己可靠的区域自主执行，人类操作员在困难状态下监控和介入；系统再根据近似 human trust 对部署数据进行重加权，用 weighted behavior cloning 更新策略。这篇和数据飞轮的关系非常直接：它不是只收离线 demonstration，而是在部署过程中持续收集 intervention data，并用这些数据提高策略自主性、降低未来人类负担。",
    "数据量": "实验覆盖仿真和真实硬件的 contact-rich manipulation tasks。数据来源包括机器人自主执行片段、人类介入片段和部署过程中产生的任务轨迹。论文重点不是发布大规模公开数据集，而是提出一套 deployment-time data collection + intervention reweighting 的学习流程。真实硬件实验体现了在任务执行中收集人类干预数据并迭代更新策略的过程。",
    "评价指标": "报告 Sirius 在仿真中相比 state-of-the-art 方法成功率提升约 8%，在真实硬件上提升约 27%；同时达到约 2 倍更快收敛和约 85% memory size reduction。指标包括 policy success rate、convergence speed、人类干预负担以及不同数据加权策略的效果。",
    "发布时间": "2022/11/15",
    "分级": "核心补充",
  },
  {
    "序号": 116,
    "论文名称": "Multi-Task Interactive Robot Fleet Learning with Visual World Models",
    "链接": "https://arxiv.org/abs/2410.22689",
    "Publication": "CoRL 2024",
    "Type": "机器人群体部署学习 / Visual World Model / Anomaly Prediction / Human Intervention",
    "简要概括": "本文提出 Sirius-Fleet，把 Sirius 从单机器人 human-in-the-loop deployment 扩展到多任务、多机器人 fleet learning。它适合用来支撑“部署反馈闭环”和“在线监控”两条线。系统先训练 visual world model，根据过去视觉帧预测未来 latent embedding，再训练 anomaly predictor 判断当前动作是否可能导致异常；当机器人策略不可靠时请求人类纠正，随着策略能力提升，异常阈值自适应调整，人类干预需求逐渐下降。对综述来说，这篇说明 data flywheel 不只是单机器人补采数据，而可以扩展到 fleet-level deployment：机器人群体在不同环境和任务中持续产生数据，模型用异常预测和人类纠正选择最有价值的反馈。",
    "数据量": "在 RoboCasa 仿真 benchmark 和 Mutex 真实世界 benchmark 上验证。数据包括多任务机器人执行轨迹、视觉观测、未来状态预测目标、人类纠正和异常标签。论文强调 large-scale multi-task setting，覆盖 household / industrial 风格多任务场景；具体使用 visual world model 预训练数据和部署中交互数据共同驱动 fleet learning。",
    "评价指标": "主要评价多任务 policy performance、monitoring accuracy、人类干预次数/工作量、异常预测准确性以及随着训练迭代机器人自主性提升的趋势。论文结论是 Sirius-Fleet 能提升多任务策略表现和监控准确率，并能在机器人能力提高后减少人类干预。",
    "发布时间": "2024/10/30",
    "分级": "核心补充",
  },
  {
    "序号": 117,
    "论文名称": "Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision",
    "链接": "https://arxiv.org/abs/2206.14349",
    "Publication": "CoRL 2022 Oral",
    "Type": "Interactive Fleet Learning / Scalable Human Supervision / On-policy Data Aggregation",
    "简要概括": "本文是机器人 fleet learning 和 scalable human supervision 的基础工作之一。它把传统 DAgger/interactive imitation learning 扩展到多个机器人和多个人类远程监督者的场景，提出 Interactive Fleet Learning (IFL) 问题设置，并提出 Return on Human Effort (ROHE) 衡量有限人力监督的收益。对具身数据飞轮综述来说，这篇非常适合作为“部署反馈不是无限人力，需要决定何时请求人类、如何分配人类注意力”的理论和实验支撑。它说明部署中的 human intervention 本身是一种稀缺数据资源，关键不只是收集干预，而是把人类反馈分配给最有价值的机器人和状态。",
    "数据量": "构建 open-source IFL benchmark suite，使用 GPU-accelerated Isaac Gym 环境进行标准化评估。仿真实验包含 100 个机器人与多个人类监督者的 fleet setting；真实实验使用 4 台 ABB YuMi 机械臂和 2 名远程人类操作员进行 image-based block pushing。数据包括机器人自主执行轨迹、人工接管/纠正数据和不同人力分配策略下的交互日志。",
    "评价指标": "提出 Return on Human Effort (ROHE) 作为核心指标，用于衡量单位人类监督带来的 fleet performance 收益。实验显示所提出的 Fleet-DAgger 算法相比 baselines 可达到最高约 8.8x ROHE；还评估 fleet task performance、human allocation strategy、人类监督利用效率和真实机器人 block-pushing 成功情况。",
    "发布时间": "2022/06/29",
    "分级": "重要",
  },
  {
    "序号": 118,
    "论文名称": "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning",
    "链接": "https://proceedings.mlr.press/v15/ross11a.html",
    "Publication": "AISTATS 2011 / PMLR",
    "Type": "DAgger / On-policy Data Aggregation / Imitation Learning 基础方法",
    "简要概括": "这是 DAgger 的基础论文，虽然不是具身大模型时代的论文，但对“数据飞轮/部署反馈”非常重要。它指出 behavior cloning 的核心问题是训练数据分布和执行时 learner induced state distribution 不匹配，错误会在长时序任务中累积。DAgger 的思想是让 learner 在自己诱导的状态分布下执行，再向 expert 查询动作，把这些状态-动作对聚合进训练集，从而形成 on-policy data aggregation。对综述来说，可以把它作为部署反馈闭环的理论源头：机器人不是只学静态专家演示，而是在自身执行产生的新状态分布上持续补数据。",
    "数据量": "论文主要是理论和算法工作，实验包括两个 imitation learning/sequential prediction 任务和一个 sequence labeling benchmark；不是机器人数据集论文。关键数据形式是迭代聚合的 expert-labeled states：每一轮用当前 policy 诱导状态分布，然后请 expert 给标签/动作，加入训练集。",
    "评价指标": "理论上给出 no-regret online learning 视角下的性能保证，解决 imitation learning 中 compounding error 和 distribution shift。实验上比较 DAgger 与 previous approaches / behavior cloning，在 sequential decision tasks 上表现更好。综述中可作为 DAgger、Fleet-DAgger、HG-DAgger 和部署中 on-policy 数据聚合的背景。",
    "发布时间": "2010/11/02",
    "分级": "借鉴",
  },
  {
    "序号": 119,
    "论文名称": "HG-DAgger: Interactive Imitation Learning with Human Experts",
    "链接": "https://arxiv.org/abs/1810.02890",
    "Publication": "ICRA 2019",
    "Type": "Human-gated DAgger / Interactive Imitation Learning / Safety-aware Human Intervention",
    "简要概括": "本文提出 HG-DAgger，解决传统 DAgger 在真实系统中让 human expert 只给标签但不真正控制系统会带来安全性和标签质量问题。HG-DAgger 让人类专家可以在认为系统不安全或不可靠时接管控制，因此数据来自 human-gated intervention，而不是被动标注。它还学习一个基于模型不确定性的风险阈值，用来预测 novice policy 在状态空间不同区域的表现。对数据飞轮综述来说，这篇可以作为“人类干预数据如何安全采集”的早期代表：部署时让人类决定何时接管，既提高安全性，也获得更高质量的 correction data。",
    "数据量": "实验包括仿真和真实世界 autonomous driving task。数据由 novice policy 执行时的人类专家接管片段、expert corrective actions 和 learner 自己诱导的状态组成。论文不是机器人 manipulation 数据集，但对 human-in-the-loop deployment 和 intervention data collection 具有方法论价值。",
    "评价指标": "比较 HG-DAgger、标准 DAgger 和 behavior cloning，指标包括任务表现、安全性、专家接管策略、模型不确定性风险阈值以及真实任务中的执行效果。论文报告 HG-DAgger 在仿真和真实 autonomous driving task 上优于 DAgger 和 behavior cloning。",
    "发布时间": "2018/10/05",
    "分级": "借鉴",
  },
  {
    "序号": 120,
    "论文名称": "REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction",
    "链接": "https://arxiv.org/abs/2306.15724",
    "Publication": "CoRL 2023 / PMLR",
    "Type": "失败解释与纠错 / RoboFail Dataset / LLM-based Failure Reasoning",
    "简要概括": "本文是失败数据和纠错数据方向的早期代表。它提出 REFLECT，把机器人多模态执行经历转成层级摘要，再交给 LLM 进行失败解释，最后由 language-based planner 根据解释生成纠正计划。核心贡献在于：失败轨迹不只是负样本，而可以被总结、解释并转化成可执行的 recovery plan。对数据飞轮综述来说，这篇可以用来说明失败数据如何进入“评测-诊断-纠错-再执行”的闭环：机器人失败后，系统不是简单丢弃轨迹，而是把失败经历转化成语言化知识和恢复策略。",
    "数据量": "构建 RoboFail 数据集，用于系统评估失败解释和纠正。数据包含 AI2-THOR 仿真中的任务执行和人工注入失败，以及真实 UR5e 机械臂遥操作产生的失败案例。公开资料显示 RoboFail 覆盖多种任务和失败场景，论文中还强调多感知输入，包括 RGB-D、audio、robot states、scene graph、event summary 和 subgoal summary。",
    "评价指标": "评估失败定位、失败解释质量和 correction planning 是否能帮助机器人完成任务。核心指标包括 LLM 生成解释的信息量、纠错计划成功率、任务最终成功情况以及与不使用 REFLECT 的 planner 对比。论文结论是基于层级经历摘要的 LLM 推理能生成有用失败解释，并辅助成功纠错。",
    "发布时间": "2023/06/27",
    "分级": "重要",
  },
  {
    "序号": 121,
    "论文名称": "AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation",
    "链接": "https://arxiv.org/abs/2410.00371",
    "Publication": "ICLR 2025",
    "Type": "VLM Failure Detection / Failure Reasoning / Synthetic Failure Data / FailGen",
    "简要概括": "本文提出 AHA，一个专门用于 robotic manipulation failure detection and reasoning 的开源 VLM。它把失败检测从二分类变成 free-form reasoning：模型不仅判断是否失败，还要用自然语言解释失败原因。论文还提出 FailGen，通过程序化扰动成功仿真 demonstrations 来生成大规模 failure trajectories，并形成 AHA dataset。对综述来说，这篇非常适合补 VLM + 失败数据 这条线：VLM 不只是做语义理解，也可以作为机器人执行过程中的 failure reasoning module，给 RL reward shaping、task planning 和 sub-task verification 提供反馈。",
    "数据量": "使用 FailGen 从 RLBench 仿真成功演示中程序化扰动关键帧，生成 AHA failure dataset；数据包含失败轨迹、自由文本失败解释和多类 failure modes。模型只在 AHA 数据上训练，但评估扩展到真实世界 failure datasets、不同机器人系统和 unseen tasks。论文还把 AHA 接入三类下游 manipulation frameworks：reinforcement learning、task and motion planning、zero-shot trajectory generation。",
    "评价指标": "AHA 在多数据集、多指标上超过第二名 GPT-4o in-context learning 10.3%，超过六个模型平均表现 35.3%。接入下游机器人系统后，AHA 的 failure feedback 通过改进 dense reward、优化 task planning、提升 sub-task verification，使三个任务的 task success rate 平均提升 21.4%。",
    "发布时间": "2024/10/01",
    "分级": "重要",
  },
  {
    "序号": 122,
    "论文名称": "SAFE: Multitask Failure Detection for Vision-Language-Action Models",
    "链接": "https://arxiv.org/abs/2506.09937",
    "Publication": "arXiv 2025",
    "Type": "VLA Failure Detection / Runtime Monitor / Conformal Prediction / Multitask Generalization",
    "简要概括": "本文直接面向 VLA 的运行时失败检测，是必须补的一篇。已有 failure detector 多数只在单任务或少数任务上训练测试，但 VLA 作为 generalist robot policy 需要在 unseen tasks 和新环境中检测失败。SAFE 的核心观察是 VLA 内部 feature space 本身已经包含关于任务成功/失败的高层信息，因此可以从 VLA internal features 学一个标量 failure likelihood，用作实时失败预警。对数据飞轮综述来说，SAFE 支撑的是部署安全和失败反馈：当 VLA 在真实环境中不可靠时，系统需要及时 stop/backtrack/ask for help，并把失败片段作为后续数据。",
    "数据量": "训练数据包含成功和失败 rollouts；评估覆盖 unseen tasks、仿真和真实世界环境。论文测试了不同 VLA policy architectures，包括 OpenVLA、π0 和 π0-FAST。数据形式是 VLA 执行过程中的内部表征、视觉观测、动作序列和成功/失败标签，用于训练 multitask failure detector。",
    "评价指标": "评价 failure detection accuracy、detection time、accuracy-time trade-off、unseen task generalization，并使用 conformal prediction 给出更可靠的报警阈值。论文报告 SAFE 相比多种 baselines 达到 state-of-the-art failure detection performance，并在检测准确性和预警时间之间取得最好 trade-off。",
    "发布时间": "2025/06/11",
    "分级": "核心补充",
  },
  {
    "序号": 123,
    "论文名称": "RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction",
    "链接": "https://arxiv.org/abs/2505.12224",
    "Publication": "arXiv 2025",
    "Type": "失败数据集 / Failure Analysis / Failure Correction / VLA外部监督",
    "简要概括": "本文提出 RoboFAC，专门解决 VLA 在 open-world manipulation 中缺乏失败恢复能力的问题。作者认为现有 VLA 主要从成功专家演示中训练，导致对失败理解和恢复能力弱。RoboFAC 同时构建失败轨迹数据集和问答数据，让模型具备 Task Understanding、Failure Analysis、Failure Correction 三类能力，并把 RoboFAC 接入真实 VLA 控制 pipeline，作为外部监督生成 correction instructions。对数据飞轮综述来说，这篇非常贴合“失败轨迹如何从负样本变成可复用监督信号”：失败数据不只是评估错误，而是能提升真实任务恢复能力。",
    "数据量": "RoboFAC dataset 包含 9,440 条错误 manipulation trajectories 和 78,623 个 QA pairs，覆盖 16 个任务、53 个场景，并同时包含仿真和真实世界环境。数据类型包括错误轨迹、视觉状态、自然语言任务、失败原因问答、纠正指令等。",
    "评价指标": "RoboFAC model 在自建 evaluation benchmark 上相比 GPT-4o 提高 34.1%。将 RoboFAC 接入真实 VLA control pipeline 后，在 4 个真实机器人任务上平均带来 29.1% relative improvement。评价维度包括 task understanding、failure analysis、failure correction 和真实任务恢复效果。",
    "发布时间": "2025/05/18",
    "分级": "重要",
  },
  {
    "序号": 124,
    "论文名称": "Yell At Your Robot: Improving On-the-Fly from Language Corrections",
    "链接": "https://arxiv.org/abs/2403.12910",
    "Publication": "arXiv 2024",
    "Type": "语言纠错数据 / On-the-fly Correction / Long-horizon Dexterous Manipulation",
    "简要概括": "本文提出 YAY Robot，关注人类能否通过自然语言纠正在机器人执行长程灵巧任务时实时帮助机器人，并让这些纠错反馈进入迭代训练。它的基本思想是：如果低层 language-conditioned skills 足够丰富，那么高层 policy 可以接受人类语言纠正，比如“move a bit to the left”，临时覆盖错误决策，并把 observation-correction pair 记录下来继续训练。对数据飞轮综述来说，这篇适合放在失败/纠错数据小节：它展示了部署中人类不需要重新遥操作整个任务，只需用语言给出局部纠错，就能产生高价值反馈数据。",
    "数据量": "实验在真实硬件上的 long-horizon dexterous manipulation tasks 中进行。数据包含原始机器人 demonstration、低层 language-conditioned skills、人类观察机器人执行时给出的语言纠正、纠正对应的视觉状态和高层策略更新样本。论文强调无需额外 teleoperation，通过 verbal feedback 形成可持续的 correction dataset。",
    "评价指标": "评价实时语言纠正后的任务完成率、迭代训练后 autonomous performance、长程任务中低层执行错误和高层决策错误的修正能力。论文报告语言反馈能显著提升真实硬件 long-horizon dexterous manipulation 任务表现，并且纠错数据可用于后续训练而不是一次性控制。",
    "发布时间": "2024/03/19",
    "分级": "重要",
  },
  {
    "序号": 125,
    "论文名称": "Don't Yell at Your Robot: Physical Correction as the Collaborative Interface for Language Model Powered Robots",
    "链接": "https://arxiv.org/abs/2412.12602",
    "Publication": "arXiv 2024 / RSS Gen-AI HRI Workshop 2024",
    "Type": "物理纠错数据 / Human Intervention / LLM-powered Robot Interface",
    "简要概括": "本文和 Yell At Your Robot 形成互补：不是用语言纠正机器人，而是让人类通过物理交互直接纠正 LLM-powered robot 的运动。机器人根据自然语言场景描述由 LLM 生成 6-DoF linear Dynamical System command；执行过程中，人类可以直接对机器人施加 physical correction，系统重新估计意图，并把修正后的 DS 转成自然语言，作为 future LLM interactions 的 prompt context。对综述来说，这篇说明 human intervention feedback 不只包含 teleoperation 和 verbal correction，还可以是更自然的物理纠正，这类数据对于真实部署的人机协作和失败恢复也很有价值。",
    "数据量": "论文提供 proof-of-concept 的 hybrid real+sim experiment。数据包括 LLM 生成的 6-DoF 动作命令、机器人执行轨迹、人类 physical correction、重新估计的 corrected DS 以及由 corrected DS 转换得到的自然语言描述。规模不大，更适合作为 interface / feedback modality 的概念性补充。",
    "评价指标": "主要是概念验证和定性/小规模实验，评估 physical correction 是否能实时改变机器人运动意图，以及修正后的动态系统能否被语言化并用于后续 prompt。适合在综述中作为 physical human feedback 的补充例子，不宜作为大规模实证主证据。",
    "发布时间": "2024/12/17",
    "分级": "借鉴",
  },
  {
    "序号": 126,
    "论文名称": "Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models",
    "链接": "https://arxiv.org/abs/2512.01946",
    "Publication": "arXiv 2025 / OpenReview 2026",
    "Type": "Failure Data Synthesis / Planning-Execution Error Detection / VLM Runtime Verification",
    "简要概括": "本文提出 Guardian，用自动合成失败数据训练 VLM 检测 robotic planning 和 execution errors。论文认为失败数据稀缺是 VLM failure detection 泛化差的核心原因，因此用程序化扰动成功轨迹来生成 planning failure 和 execution failure，并为每个失败提供细粒度类别和 step-by-step reasoning traces。对综述来说，这篇很适合放在“失败数据生成”和“在线验证”交叉处：它把成功轨迹变成失败训练数据，构造 failure reasoning dataset，再把 VLM 作为 plug-and-play verification module 接入机器人系统，用于检测失败、触发 replanning/retry 和指导恢复。",
    "数据量": "构建 FailCoT / failure reasoning 数据，基于 RLBench 仿真和 BridgeDataV2 真实机器人数据生成失败。公开资料中包含 RLBench-Fail、BridgeDataV2-Fail，并新增 UR5-Fail；数据包含多视角图像、成功/失败标签、planning/execution failure category 和 reasoning traces。项目页还显示在 RoboFail、RoboVQA 和 UR5-Fail 等 real-world benchmarks 上评估泛化。",
    "评价指标": "把 failure detection 形式化为 VQA / verification 问题，评估 planning verification、execution verification、success/failure classification、reasoning quality 和下游机器人任务成功率。论文报告 Guardian 在已有和新增 failure benchmarks 上达到 SOTA，并且接入 manipulation system 后可提升仿真和真实机器人任务成功率。",
    "发布时间": "2025/12/01",
    "分级": "前沿借鉴",
  },
  {
    "序号": 127,
    "论文名称": "Quality over Quantity: Demonstration Curation via Influence Functions for Data-Centric Robot Learning",
    "链接": "https://arxiv.org/abs/2603.09056",
    "Publication": "ICRA 2026 / arXiv 2026",
    "Type": "数据质量筛选 / Demonstration Curation / Influence Functions / Data-centric Robot Learning",
    "简要概括": "本文非常适合补“数据质量筛选/重加权”方向。它指出机器人 imitation learning 中 demonstration 质量是关键瓶颈：遥操作数据中有人类失误、操作员差异、子最优行为和噪声，单纯增加数据量可能反而引入坏动作。QoQ 的核心思想是用 influence functions 衡量每个训练样本对 validation demonstrations loss reduction 的贡献，从而自动识别高质量数据。它还针对机器人轨迹提出两个改造：用 maximum influence across validation samples 捕捉最相关状态-动作对，以及把同一轨迹内的 state-action influence 聚合，降低噪声并保持轨迹覆盖。对综述来说，这篇能把主题从“如何收集更多数据”推进到“如何筛选真正有用的数据”。",
    "数据量": "实验覆盖仿真和真实世界机器人设置。数据类型是人类遥操作 demonstration trajectories，包含质量不均、次优和有噪声的样本。论文重点不是构建新数据集，而是给出通用 curation 方法：从训练 demonstrations 中估计样本对验证示范的影响，并按轨迹层面筛选数据。",
    "评价指标": "评价 policy performance、和 prior data selection methods 的对比、仿真与真实机器人上的成功率/任务表现。论文摘要报告 QoQ 在 simulated and real-world settings 中一致提升策略表现。可在综述中作为 data-centric robot learning 的关键证据。",
    "发布时间": "2026/03/10",
    "分级": "重要",
  },
  {
    "序号": 128,
    "论文名称": "CUPID: Curating Data your Robot Loves with Influence Functions",
    "链接": "https://arxiv.org/abs/2506.19121",
    "Publication": "CoRL 2025 / arXiv 2025",
    "Type": "数据筛选 / Influence Function / Imitation Learning Policy Performance / Demonstration Curation",
    "简要概括": "本文提出 CUPID，是比通用 sample weighting 更贴近机器人 imitation learning 的数据筛选工作。它的核心问题是：如何判断每条 demonstration 对闭环 policy performance 的贡献，而不是只看离线 imitation loss。CUPID 用 influence function-theoretic formulation 估计训练 demonstration 对 expected return 的影响，从而排序、筛掉有害 demonstration，或者从新采集数据中挑选最能提升策略的数据。对数据飞轮综述来说，这篇和 QoQ 可以一起支撑“部署后采到的数据不能全量无脑加入，需要根据对闭环表现的贡献进行筛选”。",
    "数据量": "实验包括 simulated RoboMimic benchmark 和真实 Franka 硬件实验。数据形式包括 image-based diffusion policy demonstrations、不同质量的 training trajectories、新采集轨迹和 evaluation rollouts。论文报告用少于 33% 的 curated data 就可在 RoboMimic 仿真 benchmark 上达到 SOTA diffusion policy 表现，并在硬件实验中观察到类似收益。",
    "评价指标": "评价闭环 task success / expected return、数据筛选比例、对有害 demonstrations 的过滤能力、distribution shift 下 robust strategy 选择、spurious correlation 隔离以及 generalist robot policy post-training 的提升。重点不是数据规模本身，而是每条 demonstration 对 test-time performance 的可预测贡献。",
    "发布时间": "2025/06/23",
    "分级": "重要",
  },
  {
    "序号": 129,
    "论文名称": "Learning from Imperfect Demonstrations from Agents with Varying Dynamics",
    "链接": "https://arxiv.org/abs/2103.05910",
    "Publication": "ICRA 2021 / arXiv 2021",
    "Type": "Imperfect Demonstrations / Cross-dynamics Demonstration Selection / Feasibility + Optimality Score",
    "简要概括": "本文针对现实中 demonstrations 往往不是最优，或者来自动力学不同的 agent 的问题，提出如何判断一条 demonstration 对目标机器人是否有用。它用 feasibility score 衡量 demonstration 对目标 agent 是否可执行，用 optimality score 衡量 demonstration 是否接近任务最优，从而选择更有信息量的 demonstrations、忽略不相关或有害数据。对综述来说，这篇适合作为“异构数据和 imperfect demonstrations 需要筛选/重加权”的基础文献，尤其可以连接 cross-embodiment 数据：不同机器人或人类轨迹不一定都能直接用于目标 VLA，需要判断可行性和质量。",
    "数据量": "实验覆盖 4 个仿真环境和一个真实机器人实验。数据来自不同 dynamics 的 demonstrators，包括 sub-optimal demonstrations 和 dynamics mismatch demonstrations。论文重点是用 feasibility + optimality metric 对 demonstration 进行打分和筛选，而不是发布大规模数据集。",
    "评价指标": "主要评价 learned policy 的 expected return、与直接使用全部 demonstrations 或只用 optimal demonstrations 的对比、不同 dynamics mismatch 条件下的表现。实验显示利用 feasibility score 和 optimality score 选择数据能学习到更高 return 的 policy。",
    "发布时间": "2021/03/10",
    "分级": "借鉴",
  },
  {
    "序号": 130,
    "论文名称": "Learning to Weight Imperfect Demonstrations",
    "链接": "https://proceedings.mlr.press/v139/wang21aa.html",
    "Publication": "ICML 2021 / PMLR",
    "Type": "Imperfect Demonstration Weighting / GAIL / Sample Weighting 基础方法",
    "简要概括": "本文研究如何在 GAIL 中自动给 imperfect expert demonstrations 赋权。现实示范数据中常常混有不同质量样本，传统 imitation learning 容易把次优演示当成同等专家数据来学。本文提出学习示范权重，使 agent 能更依赖高质量 demonstration、降低低质量 demonstration 的负面影响。虽然这篇不是机器人 VLA 工作，但适合作为数据重加权方法的基础引用：它说明在 imitation learning 中，数据质量差异可以通过 learned weights 显式建模，而不是只依赖人工筛选。",
    "数据量": "论文主要在 imitation learning / GAIL benchmark 上评估，不是具身机器人数据集论文。数据形式是包含不同比例/不同质量 expert demonstrations 的训练集，每条 demonstration 通过学习得到权重，用于 adversarial imitation learning。",
    "评价指标": "评价 imitation policy 的任务回报/性能、对不同质量 demonstrations 的权重估计效果、与标准 GAIL 和其他 imperfect demonstration 方法的对比。可作为数据质量筛选小节中“sample weighting”方向的通用方法背景。",
    "发布时间": "2021/07/18",
    "分级": "借鉴",
  },
  {
    "序号": 131,
    "论文名称": "Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies",
    "链接": "https://arxiv.org/abs/2503.08558",
    "Publication": "RSS 2025 / arXiv 2025",
    "Type": "Runtime Failure Detection / No Failure Data / Uncertainty / Conformal Prediction",
    "简要概括": "本文提出 FAIL-Detect，解决在线失败检测中一个很现实的问题：很多系统没有足够失败数据，能否只用成功 demonstrations 预测运行时失败？作者把问题建模为 sequential OOD detection，先从 imitation learning policy 的输入输出中蒸馏出和失败相关的标量 uncertainty signals，再用 conformal prediction 做不确定性校准并给出报警阈值。对数据飞轮综述来说，这篇适合放在 online evaluation and safety monitoring：当机器人还没有收集到大量失败样本时，系统仍然需要在部署中判断何时可能失败并停止/请求帮助。",
    "数据量": "实验覆盖多种 robotic manipulation tasks，包含仿真和真实机器人场景。训练设置强调只使用 successful training data，不依赖 failure data；评估时测试 diverse and unexpected failure modes。数据包括图像观测、连续控制动作、policy inputs/outputs、uncertainty scalar signals 和 runtime trajectories。",
    "评价指标": "评价 failure detection accuracy、detection speed、与 SOTA failure detection baselines 的对比、learned scalar signals 与 post-hoc signals 的有效性、conformal prediction 校准后的可靠性。论文报告 FAIL-Detect 能比 SOTA 更准确、更早地检测失败。",
    "发布时间": "2025/03/11",
    "分级": "重要",
  },
  {
    "序号": 132,
    "论文名称": "Failure Prediction at Runtime for Generative Robot Policies",
    "链接": "https://arxiv.org/abs/2510.09459",
    "Publication": "NeurIPS 2025 / arXiv 2025",
    "Type": "Runtime Failure Prediction / Generative Imitation Learning / OOD + Action Uncertainty / Conformal Calibration",
    "简要概括": "本文提出 FIPER，面向 diffusion / flow matching 等 generative imitation learning policies 的运行时失败预测。它同样强调不需要 failure data，而是利用两个信号预测即将失败：一是 policy embedding space 中的 OOD observation，由 random network distillation 检测；二是生成动作的不确定性，用 action-chunk entropy 衡量。两个分数都用少量成功 rollouts 进行 conformal prediction 校准，并在短时间窗口内聚合触发报警。对综述来说，FIPER 可以作为 FAIL-Detect 的补充，说明随着 generative robot policies 变复杂，运行时监控需要同时看环境分布外和动作意图不确定性。",
    "数据量": "在 5 个仿真和真实环境中评估，覆盖多种 failure modes。训练/校准不需要失败样本，只使用少量 successful rollouts。数据包括生成式策略的视觉 embedding、动作块、action uncertainty、OOD scores 和真实执行轨迹。",
    "评价指标": "评价 failure prediction accuracy、early prediction time、benign OOD 与真正失败的区分能力、与已有方法的对比、conformal calibration 后报警可靠性。论文报告 FIPER 比现有方法更准确、更早预测失败，并更好地区分无害新颖状态和真实失败风险。",
    "发布时间": "2025/10/10",
    "分级": "前沿借鉴",
  },
  {
    "序号": 133,
    "论文名称": "SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning",
    "链接": "https://arxiv.org/abs/2503.03480",
    "Publication": "NeurIPS 2025 Spotlight / arXiv 2025",
    "Type": "VLA Safety Alignment / Constrained Learning / Safe RL / Long-horizon Mobile Manipulation",
    "简要概括": "本文提出 SafeVLA / Integrated Safety Approach (ISA)，是 VLA 安全对齐方向的代表。它认为 VLA 在真实部署中不仅会失败，还可能伤害环境、机器人和人，因此需要把 safety constraints 显式整合进 VLA 策略优化。方法上，它系统建模安全需求，主动 eliciting unsafe behaviors，再用 constrained Markov decision process / safe reinforcement learning 从 min-max 视角约束 VLA policy。对综述来说，这篇适合放在 deployment safety 和 online monitoring 后面：数据飞轮不能只优化成功率，还要在部署闭环中处理长尾 unsafe behaviors。",
    "数据量": "论文提出新的 benchmark environment 和安全相关数据/模型，面向 long-horizon mobile manipulation tasks。数据包括安全/不安全行为、OOD perturbations、长程移动操作轨迹、安全约束成本以及 task success 信息。项目页还释放 data、models 和 benchmark。",
    "评价指标": "核心评价 safety-performance trade-off。论文报告相比 state-of-the-art 方法，累计 safety violation cost 降低 83.58%，同时 task success rate 维持并提升 3.85%；还评估长尾风险缓解、极端失败场景、安全行为在 OOD perturbations 和 unseen tasks 上的泛化。",
    "发布时间": "2025/03/05",
    "分级": "重要",
  },
];

const sourceRows = [
  ["Robot-Powered Data Flywheels", "https://arxiv.org/abs/2511.19647"],
  ["Robot Learning on the Job / Sirius", "https://arxiv.org/abs/2211.08416"],
  ["Sirius-Fleet", "https://arxiv.org/abs/2410.22689"],
  ["Fleet-DAgger", "https://arxiv.org/abs/2206.14349"],
  ["DAgger", "https://proceedings.mlr.press/v15/ross11a.html"],
  ["HG-DAgger", "https://arxiv.org/abs/1810.02890"],
  ["REFLECT", "https://arxiv.org/abs/2306.15724"],
  ["AHA", "https://arxiv.org/abs/2410.00371"],
  ["SAFE", "https://arxiv.org/abs/2506.09937"],
  ["RoboFAC", "https://arxiv.org/abs/2505.12224"],
  ["Yell At Your Robot", "https://arxiv.org/abs/2403.12910"],
  ["Don't Yell at Your Robot", "https://arxiv.org/abs/2412.12602"],
  ["Guardian", "https://arxiv.org/abs/2512.01946"],
  ["Quality over Quantity", "https://arxiv.org/abs/2603.09056"],
  ["CUPID", "https://arxiv.org/abs/2506.19121"],
  ["Learning from Imperfect Demonstrations from Agents with Varying Dynamics", "https://arxiv.org/abs/2103.05910"],
  ["Learning to Weight Imperfect Demonstrations", "https://proceedings.mlr.press/v139/wang21aa.html"],
  ["FAIL-Detect", "https://arxiv.org/abs/2503.08558"],
  ["FIPER", "https://arxiv.org/abs/2510.09459"],
  ["SafeVLA", "https://arxiv.org/abs/2503.03480"],
];

function col(n) {
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function writeSheet(sheet, headers, data, widths) {
  const values = [headers, ...data.map((row) => headers.map((h) => row[h] ?? ""))];
  sheet.getRange(`A1:${col(headers.length)}${values.length}`).values = values;
  const used = sheet.getRange(`A1:${col(headers.length)}${values.length}`);
  used.format.font = { name: "Microsoft YaHei", size: 10, color: "#111827" };
  used.format.verticalAlignment = "top";
  used.format.wrapText = true;
  used.format.borders = { preset: "inside", style: "thin", color: "#E5E7EB" };
  const header = sheet.getRange(`A1:${col(headers.length)}1`);
  header.format.fill = "#1F4E79";
  header.format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
  header.format.horizontalAlignment = "center";
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  headers.forEach((h, i) => {
    sheet.getRange(`${col(i + 1)}1:${col(i + 1)}${values.length}`).format.columnWidthPx = widths[h] ?? 150;
  });
}

const workbook = Workbook.create();
const sheet = workbook.worksheets.getOrAdd("补充文献", { renameFirstIfOnlyNewSpreadsheet: true });
writeSheet(sheet, headers, rows, {
  "序号": 70,
  "论文名称": 420,
  "链接": 300,
  "Publication": 170,
  "Type": 300,
  "简要概括": 760,
  "数据量": 620,
  "评价指标": 620,
  "发布时间": 130,
  "分级": 110,
});

sheet.getRange(`J2:J${rows.length + 1}`).conditionalFormats.add("containsText", {
  text: "核心",
  format: { fill: "#D1FAE5", font: { color: "#065F46", bold: true } },
});
sheet.getRange(`J2:J${rows.length + 1}`).conditionalFormats.add("containsText", {
  text: "重要",
  format: { fill: "#DBEAFE", font: { color: "#1D4ED8", bold: true } },
});

const src = workbook.worksheets.add("来源");
src.getRange(`A1:B${sourceRows.length + 1}`).values = [["论文", "来源链接"], ...sourceRows];
src.getRange(`A1:B${sourceRows.length + 1}`).format.font = { name: "Microsoft YaHei", size: 10 };
src.getRange("A1:B1").format.fill = "#1F4E79";
src.getRange("A1:B1").format.font = { name: "Microsoft YaHei", size: 10, color: "#FFFFFF", bold: true };
src.getRange("A:A").format.columnWidthPx = 420;
src.getRange("B:B").format.columnWidthPx = 520;
src.freezePanes.freezeRows(1);

await fs.mkdir(outputDir, { recursive: true });
for (const sheetName of ["补充文献", "来源"]) {
  await workbook.render({ sheetName, range: "A1:J20", scale: 1 });
}
const err = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
if (/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A/.test(err.ndjson ?? "")) {
  throw new Error(`Workbook error scan found issues: ${err.ndjson}`);
}
const out = await SpreadsheetFile.exportXlsx(workbook);
await out.save(outputPath);
console.log(JSON.stringify({ outputPath, rows: rows.length }, null, 2));
