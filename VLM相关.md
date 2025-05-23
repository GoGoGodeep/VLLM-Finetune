## 视觉大模型的微调

### 一、**参数高效微调（PEFT）**  
#### 概念介绍：
这类方法通过最小化调整参数的数量和计算成本，实现模型性能提升，尤其适合资源受限的场景：  
| 方法              | 核心机制|
|-------------------|------------------------------------------------------------------------------------------|
| **LoRA**              | 在模型的权重矩阵中引入低秩矩阵，仅训练这些小矩阵而不更新原始参数。例如，在视觉语言模型中，LoRA可针对注意力机制模块（如`q_proj`、`v_proj`等）进行适配。 |
| **QLoRA**             | 结合LoRA与4-bit量化技术，将模型参数压缩后反量化训练，适用于超大规模模型的高效微调。|
|**Adapter Tuning**     | 在模型层间插入小型神经网络模块，仅训练这些模块。在微调时，除了Adapter的部分，其余的参数都是被冻住的（freeze）。|
| **Prefix Tuning**     | 前缀微调是一种轻量级的微调方法，受提示（Prompting）的启发，但它引入了可训练的连续前缀向量，作为任务特定的参数。这些前缀向量被添加到输入序列的前面，模型在生成时可以将其视为“虚拟的”提示。  

#### 相关论文/推荐阅读：
- [LORA](https://arxiv.org/pdf/2106.09685)

#### 微调尝试：
- [Qwen2-VL](https://github.com/GoGoGodeep/VLM-Finetune/tree/main/Qwen2-VL)：基于LoRA进行微调

---

### 二、**监督微调（SFT）与混合训练**  
#### 概念介绍：
| 方法              | 核心机制|
|-------------------|------------------------------------------------------------------------------------------|
| **传统监督微调**                  | 直接使用标注数据对模型进行端到端训练，但需要大量高质量数据。例如，微调视觉问答任务时需标注的（图像-问题-答案）三元组。   |
| **迭代监督微调（Iterative SFT）** | 分阶段调整模型，逐步融合视觉与文本表征。昆仑万维的R1V模型结合迭代SFT与GRPO强化学习，提升跨模态推理能力。               |

---

### 三、**强化学习微调（RLHF/RFT）** 
#### 概念介绍：
| 方法              | 核心机制|
|-------------------|------------------------------------------------------------------------------------------|
| **人类反馈强化学习（RLHF）**   | 通过奖励模型（RM）对齐人类偏好。例如，阿里Qwen2.5-VL-32B通过强化学习优化回答的准确性和人类偏好对齐，在数学推理和视觉解析任务中表现优异。    |
| **基于规则的奖励机制**         | 如DeepSeek的Visual-RFT方法，针对视觉任务（目标检测、细分类）设计可验证奖励（如IoU交并比、分类准确率），通过强化学习优化模型。              |

#### 相关论文/推荐阅读：
- [Visual-RFT](https://arxiv.org/abs/2503.01785)
