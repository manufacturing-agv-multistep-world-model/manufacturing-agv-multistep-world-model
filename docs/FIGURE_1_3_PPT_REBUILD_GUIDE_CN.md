# 图1与图3的PPT重绘指南

本指南用于在 PowerPoint 中进一步精修图1和图3。当前 Python 版本已经能够用于投稿；PPT重绘属于可选的视觉升级，不应改变框架、术语或结论边界。

## 统一画布与样式

- 页面宽度：18.3 cm，适配双栏通栏图。
- 图1建议高度：10.5 cm；图3建议高度：13.0 cm。
- 字体：Arial。一级标题 9 pt 加粗，框内主文字 7.0-7.5 pt，辅助文字 6.0-6.5 pt，子图标签 9 pt 加粗。
- 线宽：主框 1.2 pt，箭头 1.5 pt，辅助线 0.8 pt。
- 圆角矩形圆角统一；不要使用阴影、渐变、立体效果或装饰性图标。
- 主色：深蓝 `#18324A`，蓝 `#3B6EA8`，青绿 `#2A8C82`，橙 `#D4863B`，红 `#C45A4A`，灰 `#7A8793`。
- 浅底色：浅蓝 `#EAF1F7`，浅青 `#E8F4F1`，浅橙 `#FAF0E5`，浅灰 `#F3F5F7`。
- 所有文字框设置内部边距 0.08-0.12 cm；同一行对象使用“垂直居中”和“横向分布”。

## 图1：多步世界模型决策框架

### 总体结构

图1分为上下两部分。上半部分为三张等高卡片，展示“制造系统孪生 → 配对多步世界模型 → 选择性决策支持”；下半部分为四级证据与权限阶梯。

### 子图a：三张主卡片

1. 左卡片标题：`Manufacturing-system twin`。
2. 左卡片中央用一条简化正交路线和三个小圆点表示路网及车辆，不需要画完整场景。
3. 左卡片底部两行：`20-node route graph | 3 AGVs`；`Kinematics · handling · battery · charging · path-resource contention`。
4. 中卡片标题：`Paired multistep world model`。
5. 中卡片左侧放两个胶囊：蓝色 `A0`、橙色 `Ac`；两条线进入同一个 `physics-graph backbone` 方框。
6. 中卡片右下放三个青绿色胶囊：`120 s`、`360 s`、`720 s`，上方写 `paired effects`。
7. 右卡片标题：`Selective decision support`。
8. 右卡片依次写：`rank candidates`、`ensemble agreement + utility margin`、`hard safety + cooldown checks`。
9. 右卡片底部放白底青边胶囊 `shadow recommendation`，再放浅红底文字 `fallback: retain DT-aware action`。
10. 三张卡片之间只用两根粗蓝箭头连接，不再增加说明句。

### 子图b：证据与权限阶梯

按水平方向放四个编号圆点和三根浅灰箭头：

1. `Physics factorial`，副标题 `multistep fidelity`。
2. `Unseen ranking`，副标题 `counterfactual regret`。
3. `Non-acting shadow`，副标题 `coverage + benefit`。
4. `Bounded authority`，副标题 `abstain or fallback`。

前三个编号圆点用蓝色，第四个用橙色，表示只有最后一级涉及受限权限。

## 图3：详细世界模型架构

### 总体结构

图3纵向分为三个区块，分别对应“表征与滚动预测”“配对反事实推断”“选择性权限门”。三个区块之间保留明显空白，不用外框包住整张图。

### 子图a：物理感知图状态骨干

从左到右放四组对象：

1. `Graph-state inputs`：动态特征写 `occupancy · AGV · SOC · jobs`，静态特征写 `geometry · distance · capacity`。
2. `Edge-conditioned graph attention`：画四个节点和若干边，突出物理边属性进入注意力。
3. `Residual multistep rollout`：用四张轻微错位的卡片表示 `t+1`、`t+5`、`t+10`，下方写 `shared transition weights`。
4. `Prediction heads`：四条胶囊分别写 `AGV / node state`、`time + energy`、`tasks + queue`、`charge risk`。

### 子图b：冻结配对反事实推断

1. 左侧方框：`State S(t) + frozen arrivals`。
2. 分成蓝色 `baseline action A0` 和橙色 `candidate action Ac` 两条分支。
3. 两条分支进入同一个浅青色框：`Frozen V13 backbone`、`shared weights, applied twice`、`336,748 total parameters`。
4. 中间用深蓝椭圆或减号表示两次输出之差，标记 `Δ`。
5. 右侧浅橙框：`Trainable paired head`、`shared MLP`、`56,457 parameters`。
6. 最右侧放 `120 s`、`360 s`、`720 s` 三个胶囊，上方写 `paired effects`，下方写 `energy · tasks · charge queue`。

### 子图c：选择性权限门

从左到右放四个等高方框：

1. `Rank candidates`，副标题 `normalized utility`。
2. `Agreement gate`，副标题 `3 seeds + margin`。
3. `Safety gate`，副标题 `hard rules + budget`。
4. `Shadow advice`，副标题 `recommend or abstain`。

在最后一个方框下方放浅红底文字 `fallback: retain DT-aware action`。不要画成“模型直接控制车辆”，以免超出实验支持范围。

## 导出与检查

- 首选导出 PDF，并勾选高质量打印；同时保留可编辑 PPTX。
- 如需位图，导出 600 dpi TIFF 或 PNG，不要截图。
- 缩放到最终论文宽度后，最小文字仍应清晰可读。
- 检查所有箭头端点是否真正连接到框边，文字是否位于框内，英文术语是否与论文图注完全一致。
- 图内只保留结构标签和必要数值；完整解释放在图注中。
