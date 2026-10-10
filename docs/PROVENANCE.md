# 来源与修改记录

- 维护者：Yifan Ouyang / yifan-2004。
- 上游代码：https://github.com/tether-ai-research/qvac-genesis-III
- 代码快照：`86da2625744161466ba7bd6323324b0e8758d45e`。
- 筛选时 GitHub API 实时星数：0（2026-10-09 America/Los_Angeles）。星数会改变。
- 原模型：https://huggingface.co/qvac/qvac-genesis-iii-qwen3-1.7b-combined
- 原模型 revision：`39a371617fabe508eb793c9b7832ec3496d4e96e`。
- 论文：https://arxiv.org/abs/2609.19513
- 原训练机构：Tether AI Research；论文作者署名见根目录 CITATION.cff。

本发行版改写中文说明，新增纯文本补全入口、输入示例、配置说明与权重 manifest，将上游评估代码和图示移至 research/。上游评估包、架构、模型权重与 tokenizer 不改名或修改。代码采用清晰标注来源的源码快照，不包含上游 Git 历史。

这是重新整理及再分发工作，没有训练、微调、量化或合并权重。没有重新运行推理或论文评估。论文指标全部来自原模型卡，不能归为维护者自测。

原README存于 UPSTREAM_README.md，按原样归档；它的相对链接应在上游仓库阅读。原模型卡存于 UPSTREAM_MODEL_CARD.md，其图片在上游模型仓库可见。原作者图示不是本发行版实测结果。

代码使用根目录 Apache-2.0 LICENSE。模型权重许可证另存 MODEL_LICENSE；原 ATTRIBUTIONS.md 完整保留为 MODEL_ATTRIBUTIONS.md。模型卡明确将权重单独按 Apache-2.0 许可；训练数据为 CC-BY-NC 4.0，此发行版不分发训练数据。
