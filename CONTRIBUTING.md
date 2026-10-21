# 协作方式（简化版）
采用：**我先搭完整框架 → 你们各自 fork → 改自己那块 → push 到自己 fork → 口头汇报**。

## 流程

1. 我把完整框架推到 `main`。
2. 你们各自点 GitHub 的 **Fork**，然后 clone 自己的 fork：
   ```bash
   git clone https://github.com/<你的用户名>/AY10.git
   cd AY10
   ```
3. 在自己的仓库里，**只改自己负责的那几个文件**（分工见 `docs/tasks.md`）。
4. 每完成一个模块就提交、push 到**自己的 fork**：
   ```bash
   python -m pytest            # 先确认没弄坏东西
   git add .
   git commit -m "完成 xxx 模块"
   git push
   ```
5. 群里或开会时口头说明：改了哪个文件、做了什么、跑出来什么结果。
6. 最后由我按文件把你们的改动搬回主仓库（因为分工是按文件切的，基本不会冲突）。

## 三条规矩

1. **只改自己负责的目录**。要动公共文件（`src/config.py`、`src/factory.py`、
   `src/agents/base.py`、`src/env/sumo_env.py`、`tests/`）先说一声。
2. **push 前跑一次 `python -m pytest`**，全绿再 push。
3. 不要提交 `.venv/`、`results/*.csv`、`__pycache__/`（`.gitignore` 已排除，别用 `git add -f`）。

## 环境

```bash
pip install -r requirements.txt
python scripts/check_env.py       
python scripts/demo_fixed_time.py  
```

详细步骤见 `README.md` 的「环境配置」一节。

## 关于 CI（可以不管）

push 之后 GitHub 会自动跑一次测试（`.github/workflows/ci.yml`）：

- **绿勾** → 没弄坏东西；
- **红叉** → 代码有问题，点进去看报错信息。

不想用可以直接删掉 `.github/` 文件夹，**不影响任何功能**。

## 常见问题

| 报错 | 原因 / 处理 |
| --- | --- |
| `UnicodeDecodeError: 'gbk'` | `requirements.txt` 必须保持纯 ASCII，不要加中文 |
| `ImportError: SUMO_HOME` | 没设环境变量，或设置后没重开终端 |
| 开了 GUI 但没窗口 | 环境里有 `LIBSUMO_AS_TRACI=1`，libsumo 不支持 GUI |
| 状态好像没变化 | 观测已被归一化到 [0,1]，分箱阈值要用 `(0.1, 0.3, 0.6, 0.9)` |
