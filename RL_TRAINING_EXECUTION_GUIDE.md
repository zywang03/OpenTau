# OpenTau RL Training 执行策略指南

本文档基于 `/data/OpenTau/docs/source/tutorials/RECAP.rst` 提供详细的脚本执行策略，精确到具体文件路径。

## 整体流程概览

```
Stage 1: SFT训练VLA Policy (pi0)
    ↓
Stage 2: 训练Value Function
    ↓
Stage 3: Offline RL循环 (重复1-3次)
    ├─ Sub-stage 1: Rollout收集数据
    ├─ Sub-stage 2: Fine-tune Value Function
    ├─ Sub-stage 3: 计算Advantage和分位数
    └─ Sub-stage 4: Fine-tune VLA Policy
```

---

## Stage 1: SFT训练VLA Policy

### 目标
在完整的libero数据集上训练pi0 policy，达到约80%成功率（在moka pot libero-10任务上）。

### 配置文件
创建配置文件，例如：`/data/OpenTau/configs/sft_pi0_config.json`

关键配置项：
```json
{
    "dataset_mixture": {
        "datasets": [{"repo_id": "physical-intelligence/libero"}],
        "weights": [1.0],
        "action_freq": 10.0,
        "image_resample_strategy": "nearest",
        "vector_resample_strategy": "nearest"
    },
    "policy": {
        "type": "pi0",
        "pretrained_path": "lerobot/pi0",
        "advantage": "on",
        "n_obs_steps": 1,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD"
        },
        "tokenizer_max_length": 100
    }
}
```

### 执行命令

**脚本文件**: `/data/OpenTau/src/opentau/scripts/train.py`

**执行方式**:
```bash
cd /data/OpenTau
opentau-train \
    --accelerate-config=<path/to/accelerate_config.yaml> \
    --config_path=configs/sft_pi0_config.json
```

或者直接使用Python：
```bash
cd /data/OpenTau
accelerate launch \
    --config_file <path/to/accelerate_config.yaml> \
    src/opentau/scripts/train.py \
    --config_path=configs/sft_pi0_config.json
```

### 预期结果
- 训练约10k步
- 在moka pot libero-10任务上达到80%成功率
- 输出checkpoint保存在 `outputs/train/<timestamp>_pi0/checkpoints/`

---

## Stage 2: 训练Value Function

### 目标
在完整的libero数据集上训练value function，达到接近100%的准确率。

### 配置文件
创建配置文件，例如：`/data/OpenTau/configs/value_pretrain_config.json`

参考示例：`/data/OpenTau/configs/examples/value_config.json`

关键配置项：
```json
{
    "dataset_mixture": {
        "datasets": [{"repo_id": "physical-intelligence/libero"}],
        "weights": [1.0],
        "action_freq": 10.0,
        "image_resample_strategy": "nearest",
        "vector_resample_strategy": "nearest"
    },
    "policy": {
        "type": "value",
        "n_obs_steps": 1,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "VALUE": "MEAN_STD"
        },
        "max_state_dim": 32,
        "tokenizer_max_length": 52,
        "reward_config": {
            "number_of_bins": 201,
            "C_neg": -1000.0,
            "reward_normalizer": 1600,
            "N_steps_look_ahead": 50
        }
    },
    "loss_weighting": {"MSE": 0, "CE": 1}
}
```

### 执行命令

**脚本文件**: `/data/OpenTau/src/opentau/scripts/train.py`

**执行方式**:
```bash
cd /data/OpenTau
opentau-train \
    --accelerate-config=<path/to/accelerate_config.yaml> \
    --config_path=configs/value_pretrain_config.json
```

### 预期结果
- 训练约80k步
- 在libero数据集上达到接近100%准确率
- 输出checkpoint保存在 `outputs/train/<timestamp>_value/checkpoints/`

---

## Stage 3: Offline RL训练循环

以下步骤需要重复1-3次，直到达到期望性能。

### Sub-stage 1: Rollout收集数据

### 目标
使用当前训练好的policy在libero仿真环境中rollout，收集约300个episode（包括成功和失败的）。

### 配置文件
创建评估配置文件，例如：`/data/OpenTau/configs/rollout_config.json`

关键配置项：
```json
{
    "policy": {
        "type": "pi0",
        "pretrained_path": "<path/to/stage1_or_stage3.4_checkpoint>",
        "n_obs_steps": 1
    },
    "env": {
        "suite": "libero",
        "task": "<specific_libero_task>",
        "num_envs": 1
    },
    "eval": {
        "num_episodes": 300,
        "save_videos": true,
        "save_rollouts": true
    }
}
```

### 执行命令

**脚本文件**: `/data/OpenTau/src/opentau/scripts/eval.py`

**执行方式**:
```bash
cd /data/OpenTau
opentau-eval \
    --accelerate-config=<path/to/accelerate_config.yaml> \
    --config_path=configs/rollout_config.json
```

或者直接使用Python：
```bash
cd /data/OpenTau
accelerate launch \
    --config_file <path/to/accelerate_config.yaml> \
    src/opentau/scripts/eval.py \
    --config_path=configs/rollout_config.json
```

### 预期结果
- 收集约300个episode数据
- 数据保存在指定路径（例如：`<path/to/libero-rollouts>`）
- 数据格式符合LeRobot dataset格式

---

### Sub-stage 2: Fine-tune Value Function

### 目标
在原始数据集和所有之前收集的rollout数据集上fine-tune value function。

### 配置文件
创建配置文件，例如：`/data/OpenTau/configs/value_finetune_config.json`

关键配置项：
```json
{
    "dataset_mixture": {
        "datasets": [
            {
                "repo_id": "physical-intelligence/libero",
                "episodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            {
                "repo_id": "OpenTau/libero-rollouts",
                "root": "<path/to/libero-rollouts>"
            }
        ],
        "weights": [1.0, 1.0],
        "action_freq": 10.0,
        "image_resample_strategy": "nearest",
        "vector_resample_strategy": "nearest"
    },
    "policy": {
        "type": "value",
        "pretrained_path": "<path/to/stage2_checkpoint>",
        "n_obs_steps": 1,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "VALUE": "MEAN_STD"
        },
        "max_state_dim": 32,
        "tokenizer_max_length": 52,
        "reward_config": {
            "number_of_bins": 201,
            "C_neg": -1000.0,
            "reward_normalizer": 1600,
            "N_steps_look_ahead": 50
        }
    },
    "loss_weighting": {"MSE": 0, "CE": 1}
}
```

**注意**: 使用Stage 2的pretrained checkpoint作为起点，而不是上一轮迭代的value function。

### 执行命令

**脚本文件**: `/data/OpenTau/src/opentau/scripts/train.py`

**执行方式**:
```bash
cd /data/OpenTau
opentau-train \
    --accelerate-config=<path/to/accelerate_config.yaml> \
    --config_path=configs/value_finetune_config.json
```

### 预期结果
- Fine-tuned value function checkpoint
- 输出保存在 `outputs/train/<timestamp>_value/checkpoints/`

---

### Sub-stage 3: 计算Advantage和分位数

### 目标
使用fine-tuned value function计算每个数据点的advantage，并计算epsilon阈值（使得30%数据点有正advantage，70%有负advantage）。

### 配置文件

**数据集配置文件**: `/data/OpenTau/configs/examples/advantage_config.json`

或者创建新的：`/data/OpenTau/configs/advantage_calc_config.json`

```json
{
    "datasets": [
        {
            "repo_id": "physical-intelligence/libero",
            "root": "<path/to/libero>",
            "episodes": [0,1,2,3,4,5,6,7,8,9]
        },
        {
            "repo_id": "OpenTau/libero-rollouts",
            "root": "<path/to/libero-rollouts>"
        }
    ],
    "weights": [1.0, 1.0],
    "action_freq": 30.0,
    "image_resample_strategy": "nearest",
    "vector_resample_strategy": "nearest"
}
```

### 执行命令

**脚本文件**: `/data/OpenTau/src/opentau/scripts/get_advantage_and_percentiles.py`

**执行方式**:
```bash
cd /data/OpenTau
python src/opentau/scripts/get_advantage_and_percentiles.py \
    --config_path=<path/to/substage2_value_checkpoint> \
    --batch_size=20 \
    --dataloader_batch_size=20 \
    --dataset_mixture=configs/advantage_calc_config.json
```

**示例**（基于文档）:
```bash
cd /data/OpenTau
python src/opentau/scripts/get_advantage_and_percentiles.py \
    --config_path=outputs/train/2025-11-29/00-38-59_value/checkpoints/00520000 \
    --batch_size=20 \
    --dataloader_batch_size=20 \
    --dataset_mixture=configs/examples/advantage_config.json
```

### 预期结果
- Advantage值保存到每个数据集的 `advantages.json` 文件中
- 控制台输出advantage分位数（0th, 5th, 10th, ..., 100th）
- 根据分位数确定epsilon阈值（例如：70th percentile的值）

**输出示例**:
```
Advantage percentiles for deciding epsilon threshold:
  000th percentile: -0.123456
  005th percentile: -0.098765
  ...
  070th percentile: -0.012345  # <-- 这可能是epsilon阈值
  ...
  100th percentile: 0.045678
```

---

### Sub-stage 4: Fine-tune VLA Policy

### 目标
在原始数据集和所有rollout数据集上fine-tune VLA policy，使用advantage阈值设置 `I_t` indicator。

### 配置文件
创建配置文件，例如：`/data/OpenTau/configs/pi0_finetune_config.json`

关键配置项：
```json
{
    "dataset_mixture": {
        "datasets": [
            {"repo_id": "physical-intelligence/libero"},
            {
                "repo_id": "OpenTau/libero-rollouts",
                "root": "<path/to/libero-rollouts>"
            }
        ],
        "weights": [1.0, 1.0],
        "action_freq": 10.0,
        "image_resample_strategy": "nearest",
        "vector_resample_strategy": "nearest"
    },
    "policy": {
        "type": "pi0",
        "pretrained_path": "<path/to/stage1_checkpoint>",
        "advantage": "use",
        "advantage_threshold": <epsilon_threshold_from_substage3>,
        "n_obs_steps": 1,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD"
        },
        "tokenizer_max_length": 100
    }
}
```

**注意**: 
- 使用Stage 1的pretrained checkpoint作为起点，而不是上一轮迭代的policy
- `advantage_threshold` 使用Sub-stage 3计算出的epsilon值
- `advantage: "use"` 表示使用advantage进行数据筛选

### 执行命令

**脚本文件**: `/data/OpenTau/src/opentau/scripts/train.py`

**执行方式**:
```bash
cd /data/OpenTau
opentau-train \
    --accelerate-config=<path/to/accelerate_config.yaml> \
    --config_path=configs/pi0_finetune_config.json
```

### 预期结果
- Fine-tuned VLA policy checkpoint
- 输出保存在 `outputs/train/<timestamp>_pi0/checkpoints/`
- 该checkpoint可用于下一轮迭代的Sub-stage 1 rollout

---

## 完整执行流程示例

假设所有配置文件已准备好，accelerate config在 `accelerate_config.yaml`：

```bash
# ============================================
# Stage 1: SFT训练VLA Policy
# ============================================
cd /data/OpenTau
opentau-train \
    --accelerate-config=accelerate_config.yaml \
    --config_path=configs/sft_pi0_config.json

# 记录checkpoint路径
SFT_CHECKPOINT="outputs/train/$(ls -t outputs/train | head -1 | grep pi0)/checkpoints/$(ls -t outputs/train/$(ls -t outputs/train | head -1 | grep pi0)/checkpoints | head -1)"

# ============================================
# Stage 2: 训练Value Function
# ============================================
opentau-train \
    --accelerate-config=accelerate_config.yaml \
    --config_path=configs/value_pretrain_config.json

# 记录checkpoint路径
VALUE_CHECKPOINT="outputs/train/$(ls -t outputs/train | head -1 | grep value)/checkpoints/$(ls -t outputs/train/$(ls -t outputs/train | head -1 | grep value)/checkpoints | head -1)"

# ============================================
# Stage 3: Offline RL循环 (迭代1)
# ============================================

# Sub-stage 1: Rollout
opentau-eval \
    --accelerate-config=accelerate_config.yaml \
    --config_path=configs/rollout_config.json \
    --policy.pretrained_path=$SFT_CHECKPOINT

# Sub-stage 2: Fine-tune Value Function
opentau-train \
    --accelerate-config=accelerate_config.yaml \
    --config_path=configs/value_finetune_config.json \
    --policy.pretrained_path=$VALUE_CHECKPOINT

VALUE_FINETUNE_CHECKPOINT="outputs/train/$(ls -t outputs/train | head -1 | grep value)/checkpoints/$(ls -t outputs/train/$(ls -t outputs/train | head -1 | grep value)/checkpoints | head -1)"

# Sub-stage 3: 计算Advantage
python src/opentau/scripts/get_advantage_and_percentiles.py \
    --config_path=$VALUE_FINETUNE_CHECKPOINT \
    --batch_size=20 \
    --dataloader_batch_size=20 \
    --dataset_mixture=configs/advantage_calc_config.json

# 从输出中提取epsilon阈值（例如70th percentile）
EPSILON_THRESHOLD=-0.012345  # 替换为实际值

# Sub-stage 4: Fine-tune VLA Policy
opentau-train \
    --accelerate-config=accelerate_config.yaml \
    --config_path=configs/pi0_finetune_config.json \
    --policy.pretrained_path=$SFT_CHECKPOINT \
    --policy.advantage_threshold=$EPSILON_THRESHOLD

# 更新SFT_CHECKPOINT为新的fine-tuned checkpoint用于下一轮迭代
SFT_CHECKPOINT="outputs/train/$(ls -t outputs/train | head -1 | grep pi0)/checkpoints/$(ls -t outputs/train/$(ls -t outputs/train | head -1 | grep pi0)/checkpoints | head -1)"

# ============================================
# 重复Stage 3的步骤1-3次，直到达到期望性能
# ============================================
```

---

## 关键文件路径总结

| 阶段 | 脚本文件 | 配置文件示例 |
|------|---------|-------------|
| Stage 1 | `/data/OpenTau/src/opentau/scripts/train.py` | `configs/sft_pi0_config.json` |
| Stage 2 | `/data/OpenTau/src/opentau/scripts/train.py` | `configs/value_pretrain_config.json` |
| Stage 3.1 | `/data/OpenTau/src/opentau/scripts/eval.py` | `configs/rollout_config.json` |
| Stage 3.2 | `/data/OpenTau/src/opentau/scripts/train.py` | `configs/value_finetune_config.json` |
| Stage 3.3 | `/data/OpenTau/src/opentau/scripts/get_advantage_and_percentiles.py` | `configs/advantage_calc_config.json` |
| Stage 3.4 | `/data/OpenTau/src/opentau/scripts/train.py` | `configs/pi0_finetune_config.json` |

---

## 注意事项

1. **Checkpoint路径**: 每次训练后记录checkpoint路径，用于后续步骤
2. **数据集路径**: 确保rollout数据保存路径正确，并在后续配置中引用
3. **Epsilon阈值**: 从Sub-stage 3的输出中提取70th percentile作为阈值
4. **Pretrained路径**: 
   - Value function fine-tuning使用Stage 2的checkpoint（不是上一轮迭代的）
   - VLA policy fine-tuning使用Stage 1的checkpoint（不是上一轮迭代的）
5. **迭代次数**: 根据性能需求，重复Stage 3的步骤1-3次

---

## 参考文档

- 主要文档: `/data/OpenTau/docs/source/tutorials/RECAP.rst`
- Value配置示例: `/data/OpenTau/configs/examples/value_config.json`
- Advantage配置示例: `/data/OpenTau/configs/examples/advantage_config.json`
