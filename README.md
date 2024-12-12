从零开始打造一个AI雀士

~~因为在欲之间屠杀的战绩比金之间还猛被封号了，所以本着十分信任猫粮的封号能力不会造成多大后果的想法来分享一下做法~~

### 前端
1. **使用 [majsoul_wrapper](https://github.com/747929791/majsoul_wrapper)**
   - 按照教程和 issue 中给出的环境设置。
   - 将 `npm` 安装的 `protobufjs` 替换为 `protobufjs-cli`，然后继续按照教程操作。
   - 注意控制 `protobuf` 和 `protoc` 的版本以避免兼容性问题。

2. **调整代码**
   - 下载并覆盖我仓库中改动过的几个 Python 脚本。
   - 仓库中的 `myReadme.md` 提供了后续操作指南。
   - 我的主要改动包括：
     - 适配 `MORTAL`。
     - 使用 `win32api` 替代 `pyautogui`，从而实现无需将雀魂界面置顶，可以在其他桌面挂机。

### 后端
1. **使用 [MORTAL](https://github.com/Equim-chan/Mortal)**
   - **数据准备**
     - 前往贴吧搜索天凤凤凰桌 2009~2017 年的牌谱（XML 格式）。
     - 使用 [mjlog2mjai](https://github.com/fstqwq/mjlog2mjai) 将 XML 转换为 MJAI 格式的 JSON 文件。
       - 转换时直接将 JSON 文件压缩为 `json.gz` 格式，以节省存储空间（未压缩文件可能占用百倍空间）。
     - 处理 2016 至 2017 年的牌谱时可能遇到问题。实际上，使用 2010 至 2015 年的数据训练效果已足够。
     - 转换过程中若出现损坏牌谱，可使用 `mortal` 自带的 `target/release/validate_logs` 工具检测并删除损坏的文件。

2. **模型训练**
   - 修改 `Mortal/mortal` 中的示例配置文件，并将其放入 `Mortal` 目录下。
   - 执行以下步骤：
     1. `python mortal/train_grp.py` 训练至正确率约 0.22。
     2. 运行 `python mortal/train.py`：
        - 按照 issue 和讨论区中的建议，先不加载模型，使用随机模型训练。
        - 在第一个 `save_every` 保存点后，将保存的模型设为基线模型（baseline）。
        - 调整配置后继续训练 offline。通过这一流程，模型可以基本掌握人类棋手的基本招数。

3. **启动后端服务**
   - 将仓库中的 `remote.py` 放入 `Mortal/mortal` 目录。
   - 运行 `python mortal/remote.py` 启动后端服务。

### 启动完整流程
1. 启动 `majsoul_wrapper` 的网页服务，并调至繁体中文
2. 启动 `Mortal` 的后端服务
3. 运行前端脚本 `main2.py`。
