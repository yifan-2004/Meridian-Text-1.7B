# QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training

<h4 align="left">
  📄 <a href="https://arxiv.org/abs/2609.19513">Paper (arXiv)</a> · 🤗 <a href="https://huggingface.co/datasets/qvac/GenesisIII">Hugging Face Dataset</a> · 💻 <a href="https://github.com/tether-ai-research/qvac-genesis-III">GitHub</a>
</h4>

## Key highlights

- **Peer-Reviewed at COLM 2026:** The [QVAC Genesis III paper](https://arxiv.org/abs/2609.19513) from Tether AI Research was accepted at the Conference on Language Modeling (COLM) 2026, providing peer-reviewed recognition of the corpus, methodology, and controlled evaluation.
- **An Expanding Research Program:** Genesis I introduced **40.91B tokens** and Failure Analysis. Genesis II added **107.46B new tokens** and Option-Level Reasoning, bringing Genesis I and II to **148.37B tokens across 19 domains**. Genesis III adds a further **43.06B tokens**, expanding the corpus to **191.43B tokens**, and validates the combined methodology through the controlled experiments presented in the COLM 2026 paper.
- **Large-Scale STEM Coverage:** QVAC Genesis III contains **191.43 billion tokens and approximately 160 million documents** across 19 STEM domains, three difficulty levels—**high school, college, and professional**—and four educational styles.
- **Learning from Both Failures and Successes:** A dual generation strategy uses a weak student model to determine what each example should teach. Failures become targeted corrective explanations, while successes become contrastive analyses of every answer option.
- **Two Complementary Training Splits:** The corpus contains **108.67B Option-Level (OL) tokens**, generated from correctly answered questions, and **82.76B Failure Analysis (FA) tokens**, generated from incorrect or non-extractable answers.
- **Large Gains Under a Matched Token Budget:** Against a token-matched Cosmopedia-v2 run, the combined Genesis III corpus improves accuracy by **28.57 points on ARC-Easy**, **21.35 points on ARC-Challenge**, **2.52 points on GPQA Diamond**, and **15.03 points on MMLU STEM**.
- **Near-Perfect Answer Extractability:** The OL split reaches a **99.45% Valid Answer Rate (VAR)** on MMLU STEM, meaning nearly every response contains one clear, extractable final answer.
- **QVAC Genesis III** is made available by Tether AI Research under the CC-BY-NC 4.0 (Creative Commons Attribution–NonCommercial 4.0), allowing free use and adaptation for non-commercial research and educational purposes while ensuring appropriate credit to us. A model trained on the combined Option-Level (OL) and Failure Analysis (FA) tokens is also being released on the Apache 2.0 license for research and education purposes.

**Copyright Complaints:** We will take appropriate actions in response to notice of copyright infringement. If you believe your work has been used or copied in a manner that infringes upon your intellectual property rights, please email [data-apps@tether.io](mailto:data-apps@tether.io) identifying and describing both the copyrighted work and alleged infringing content to file a notice of infringement.

<div align="center" style="background:#f4f6f9; border:1px solid #e5e9ef; border-radius:12px; padding:24px 28px; margin:28px 0;">
  <h3 style="margin:0 0 8px 0; font-size:20px; color:#111;">🚀 QVAC Genesis III on Hugging Face</h3>
  <p style="margin:0 0 16px 0; color:#555;">Access the QVAC Genesis III dataset and resources in one place.</p>
  <a href="https://huggingface.co/collections/qvac/genesis-iii" style="display:inline-block; background:#1f7feb; color:#ffffff; padding:10px 22px; margin:4px; border-radius:8px; text-decoration:none; font-weight:600;">🔗 Open the Collection</a>
</div>

## 1. Introduction

High-quality STEM pre-training data remains scarce in the open ecosystem. This is particularly limiting for small language models intended for edge and on-device deployment, where both model capacity and training-token budgets are constrained.

[QVAC Genesis I](https://huggingface.co/blog/qvac/genesis-i) introduced Tether Data, S.A. de C.V.’s (Tether AI Research, we, our)  Learning from Failures approach: incorrect student answers were turned into educational explanations that diagnose the error and teach the correct solution. [QVAC Genesis II](https://huggingface.co/blog/qvac/genesis-ii) added Option-Level Reasoning, which uses correctly answered questions to explain why the correct option works and why every distractor fails.

QVAC Genesis III develops this dual approach into a 191.43B-token corpus and evaluates it through controlled from-scratch pre-training experiments. The accompanying paper, **accepted at COLM 2026**, also introduces an LLM-as-a-parser evaluation protocol that separates answer correctness from the ability to produce a clear, extractable final answer.

## 2. Building QVAC Genesis III

The generation pipeline has four stages:

1. Acquire and quality-filter domain-specific seed passages.
2. Generate self-contained multiple-choice questions.
3. Ask an edge-scale student model to answer each question and extract its final choice.
4. Route the example to Failure Analysis or Option-Level Reasoning.

![pipeline](https://cdn-uploads.huggingface.co/production/uploads/66ad47f5a45133da70d1c40b/Ha8so8hEmzOpkPWHfbMo3.png)

### 2.1 Seeds and question generation

Seed passages are sampled from [FineFineWeb](https://huggingface.co/datasets/m-a-p/FineFineWeb) and filtered with the [Ultra-FineWeb classifier](https://huggingface.co/openbmb/Ultra-FineWeb-classifier). Sampling continues until approximately 500,000 high-quality seeds remain.

[QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) then generates self-contained questions with exactly four mutually exclusive options and one gold label. The prompt enforces an even distribution of correct labels across A, B, C, and D. A lightweight format validator rejects approximately 15% of raw generations.

### 2.2 Turning student behavior into training data

The data-generation student is [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base), selected after comparison with Llama-3.2-1B, Gemma-3-1B, and SmolLM2-1.7B on the target STEM benchmarks.

The student produces a free-form answer to each generated question. CompassJudger-2-32B parses the response and returns one option label, `NO_ANSWER` when no choice is extractable, or `MULTIPLE_ANSWERS` when conflicting final choices remain. This parser extracts the student's committed choice; it does not solve or grade the question.

The extracted choice determines the route:

- **Failure Analysis (FA):** An incorrect or non-extractable response is sent to QwQ-32B together with the question, student response, and gold answer. The teacher diagnoses the likely error, explains the correction, and produces a self-contained lesson.
- **Option-Level Reasoning (OL):** A correctly answered question is sent to QwQ-32B with its options and gold answer. The teacher produces a self-contained analysis that justifies the correct option and explicitly refutes every distractor.

Both routes render content in four styles: educational textbook, web article, question-answer tutoring, and conversational dialogue.

## 3. Corpus composition

| Split | Tokens | Documents |
| :---- | ----: | ----: |
| Option-Level Reasoning | 108.67B | 92,538,646 |
| Failure Analysis | 82.76B | 67,107,907 |
| **Total** | **191.43B** | **159,646,553** |

The corpus covers 19 curriculum-aligned domains:

- Astronomy
- Electrical engineering
- High school geography
- College biology
- High school biology
- College medicine
- Professional medicine
- College mathematics
- High school mathematics
- College physics
- High school physics
- Conceptual physics
- College chemistry
- High school chemistry
- College computer science
- High school computer science
- Machine learning
- High school statistics
- Econometrics

The three difficulty levels are high school, college, and professional.

### Deduplication and decontamination

MinHash near-deduplication with 60-gram signatures flags 47,927 duplicate pairs. Of these, only **1,729 unique documents** are identified as near-duplicates, less than 0.002% of the corpus.

The corpus is also scanned against the benchmark sets included in the paper's decontamination analysis with AllenAI's Decon overlap detector. A document is marked as contaminated when answer overlap is at least 60% with meaningful term matches or passage overlap is at least 40%.

- **17 verified contaminated documents** among 159.6M Genesis III documents, compared with 313 among 39.1M Cosmopedia-v2 documents.
- All 17 Genesis III matches are in the FA split; the OL split has **zero** verified contamination.
- GSM8K test leakage is **0%**.
- MMLU test leakage is **0.005%**: 3 matches among 56,168 examples.

## 4. Evaluation: parsing, not judging

Standard multiple-choice evaluation often scores the likelihood of answer tokens without testing what the model actually generates. Genesis III instead uses a generation-first protocol built on [OpenCompass](https://github.com/open-compass/opencompass):

1. The evaluated model generates a free-form response.
2. A separate extractor model recovers the final committed option or abstains.
3. The extracted option is compared with the benchmark gold label.

![eval](https://cdn-uploads.huggingface.co/production/uploads/66ad47f5a45133da70d1c40b/yXNs5LmZd8exbbj_w6xFV.png)

We report two metrics:

- **Accuracy:** the percentage of all examples for which the extracted option matches the gold label. Missing or conflicting answers count as incorrect.
- **Valid Answer Rate (VAR):** the percentage of responses containing one unambiguous, extractable option.

Within this parser-based protocol, VAR measures answer extractability separately from correctness.

## 5. Controlled pre-training experiments

The primary controlled runs use the Qwen3-1.7B architecture initialized from random weights. Training hyperparameters are identical across runs, making data composition the primary experimental variable; token budgets are reported separately below.

The models are trained with Megatron-Core and Megatron-Bridge using Flash Attention 2, BF16 precision, a sequence length of 4,096, and packed sequences with attention resets at document boundaries. Training runs on 64 NVIDIA H100 80GB GPUs.

We evaluate on:

- ARC-Easy and ARC-Challenge
- GPQA Diamond
- The 19 MMLU STEM domains aligned with the Genesis III curriculum

These benchmarks are intentionally STEM-aligned with the training distribution. The paper excludes non-STEM MMLU domains because Genesis III does not cover them.

### 5.1 Individual split ablation

The OL and Cosmopedia-v2 runs have closely matched token budgets. The FA run is approximately 25% smaller by design.

| Model / training data | Training tokens | ARC-E | ARC-C | GPQA | MMLU STEM accuracy | MMLU STEM VAR |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| Cosmopedia-v2, 4 epochs | ~109.7B | 20.63 | 21.02 | 17.67 | 20.39 | 72.71 |
| Genesis III — FA, 1 epoch | 82.76B | 29.81 | 23.05 | 21.21 | 23.29 | 78.14 |
| Genesis III — OL, 1 epoch | 108.67B | **47.44** | **36.27** | **27.78** | **30.26** | **99.45** |
| **OL improvement over Cosmopedia-v2** |  | **+26.81** | **+15.25** | **+10.11** | **+9.87** | **+26.74** |

Both Genesis III splits outperform the same four-epoch Cosmopedia-v2 baseline on every reported benchmark. OL and Cosmopedia-v2 have closely matched token budgets; FA achieves its gains with approximately 25% fewer tokens. OL is the stronger standalone split, while FA shows that incorrect and ambiguous student responses still provide useful pre-training signal.

![genesis_split_ablation](https://cdn-uploads.huggingface.co/production/uploads/66ad47f5a45133da70d1c40b/9xYFP3QhUMrCKBuVGKPbh.png)

### 5.2 Full corpus against baselines

The combined Genesis III run uses one epoch over 191.43B tokens. Its controlled baseline is a token-matched Cosmopedia-v2 run. We also report the publicly released [Cosmo-1B](https://huggingface.co/HuggingFaceTB/cosmo-1b), an uncontrolled comparison: Cosmo-1B has 1.8B parameters and was trained on 180B tokens including additional code, mathematics, instruction-following, and chat-formatted data.

| Model / training data | Training tokens | ARC-E | ARC-C | GPQA | MMLU STEM accuracy | MMLU STEM VAR |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| Cosmopedia-v2, 7 epochs | ~192.4B | 23.28 | 21.36 | 20.20 | 15.16 | 62.48 |
| Cosmo-1B | 180B | 28.04 | 23.73 | 19.70 | 25.42 | **92.51** |
| Genesis III Combined, 1 epoch | 191.43B | **51.85** | **42.71** | **22.72** | **30.19** | 92.06 |
| **Improvement over Cosmopedia-v2** |  | **+28.57** | **+21.35** | **+2.52** | **+15.03** | **+29.58** |
| **Improvement over Cosmo-1B** |  | **+23.81** | **+18.98** | **+3.02** | **+4.77** | −0.45 |

The combined model improves accuracy over both baselines on all four benchmark groups. Its VAR is 29.58 points above the token-matched Cosmopedia-v2 run and 0.45 points below Cosmo-1B, which the paper describes as comparable. The reported results are point estimates without confidence intervals, so small differences should not be read as evidence of statistical significance.

![genesis_combined_comparison](https://cdn-uploads.huggingface.co/production/uploads/66ad47f5a45133da70d1c40b/eaGZc_nKwJcZai0JVmtOL.png)

## 6. What the ablations show

Three findings stand out:

1. **Successes and failures carry different signals.** OL outperforms FA in 18 of 19 MMLU STEM domains, while FA is stronger in Professional Medicine. On High School Geography, the full 191.43B-token combined run reaches 35.4% accuracy, compared with 31.8% for the 108.67B-token OL run and 25.8% for the 82.76B-token FA run.
2. **Structured option-level data improves answer extractability.** OL reaches 100% VAR in 15 of 19 MMLU STEM domains and 99.45% overall on MMLU STEM. Its conflicting-answer rate is only 0.06%.
3. **The gains transfer across architectures.** Repeating the OL-versus-Cosmopedia-v2 comparison with Llama-3.2-1B, SmolLM2-1.7B, and Gemma-3-1B improves every reported metric for every tested backbone.

![genesis_cross_architecture](https://cdn-uploads.huggingface.co/production/uploads/66ad47f5a45133da70d1c40b/_ZWQYc7wYG4wMQnPpjtML.png)

Additional checks support the main result:

- At the closest same-stage checkpoint, corresponding to approximately 25B Cosmopedia-v2 tokens, OL improves over Cosmopedia-v2 by 20.80 points on MMLU STEM, 15.93 on ARC-C, 16.94 on ARC-E, and 8.08 on GPQA Diamond.
- Re-parsing saved MMLU predictions with GPT-OSS-20B and GPT-OSS-120B preserves OL's VAR advantage over the seven-epoch Cosmopedia-v2 endpoint. Across CompassJudger-2-32B and the two independent extractors, that advantage is 24.28–36.97 points.

## 7. Conclusion

QVAC Genesis III is a 191.43B-token synthetic STEM corpus designed to make each pre-training token more useful for small, efficient language models. Its dual pipeline turns failures into corrections and successes into exhaustive option-level explanations. Controlled from-scratch experiments show consistent gains over Cosmopedia-v2, while the separate, uncontrolled Cosmo-1B comparison shows higher Genesis III accuracy on the reported benchmarks. The LLM-as-a-parser protocol makes response validity measurable rather than implicit.

Together with [Genesis I](https://huggingface.co/blog/qvac/genesis-i) and [Genesis II](https://huggingface.co/blog/qvac/genesis-ii), Genesis III continues our work on open pre-training data and models for education and STEM reasoning, with Genesis III specifically targeting efficient edge and on-device models.

## References

1. QVAC. [QVAC Genesis I: the Largest and Highest-Quality Multi-domain Educational Synthetic Dataset for Pre-training](https://huggingface.co/blog/qvac/genesis-i).
2. QVAC. [QVAC Genesis II: Expanding the Largest and Highest-Quality Multi-domain Educational Synthetic Dataset for LLM Pre-training](https://huggingface.co/blog/qvac/genesis-ii).
3. Hugging Face. [Cosmopedia: How to Create Large-Scale Synthetic Data for Pre-training](https://huggingface.co/blog/cosmopedia).
4. M-A-P. [FineFineWeb](https://huggingface.co/datasets/m-a-p/FineFineWeb).
5. Wang et al. [Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data](https://arxiv.org/abs/2505.05427).
6. OpenCompass Contributors. [OpenCompass](https://github.com/open-compass/opencompass).
7. Clark et al. [Think You Have Solved Question Answering? Try ARC](https://arxiv.org/abs/1803.05457).
8. Rein et al. [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022).
9. Hendrycks et al. [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300).

## Citation

If you use QVAC Genesis III in your work, please cite:

```bibtex
@misc{vitabile2026qvacgenesisiii,
  title         = {QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training},
  author        = {Davide Vitabile and Nikhil Ranjan and Akshay Nambiar and Kamal Kumar Gupta and Amril Nazir},
  year          = {2026},
  eprint        = {2609.19513},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  institution   = {Tether Data, S.A. de C.V. d.b.a. Tether AI Research},
  note          = {Accepted at the Conference on Language Modeling (COLM) 2026},
  url           = {https://arxiv.org/abs/2609.19513}
}
```



Licensing Information: This model which was trained as described above is licensed by Tether Data, S.A. de C.V. under the Apache 2.0 license. As described above, this model is a version of the Qwen3-1.7B (https://huggingface.co/Qwen/Qwen3-1.7B) which is licensed under the Apache 2.0 license.
As described above, the Qwen3-1.7B was initialized from random weights and trained using the combined Option-Level (OL) and Failure Analysis (FA) tokens from the Genesis III dataset (https://huggingface.co/datasets/qvac/GenesisIII) which is licensed under the CC-BY-NC 4.0 (Creative-Commons-Attribution-Non Commercial 4.0). 


Licenses are reproduced below.

## **Apache License**

Version 2.0, January 2004  
[http://www.apache.org/licenses/](http://www.apache.org/licenses/)  
   
TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. **Definitions.**  
   "**License**" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.

   "**Licensor**" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.

   "**Legal Entity**" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "**control**" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

   "**You**" (or "**Your**") shall mean an individual or Legal Entity exercising permissions granted by this License.

   "**Source**" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.

   "**Object**" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.

   "**Work**" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

   "**Derivative Works**" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

   "**Contribution**" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "**submitted**" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "**Not a Contribution**."

   "**Contributor**" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.

2. **Grant of Copyright License**. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.  
3. **Grant of Patent License**. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.  
4. **Redistribution**. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:

   (a)   You must give any other recipients of the Work or Derivative Works a copy of this License; and

   (b)   You must cause any modified files to carry prominent notices stating that You changed the files; and

   (c)   You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and

   (d)   If the Work includes a "**NOTICE**" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License.

   You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

5. **Submission of Contributions**. Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.  
6. **Trademarks**. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.  
7. **Disclaimer of Warranty**. Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.  
8. **Limitation of Liability**. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.  
9. **Accepting Warranty or Additional Liability.** While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

**CC BY-NC 4.0**  
**Attribution-NonCommercial 4.0 International**  
[https://creativecommons.org/licenses/by-nc/4.0/legalcode.en](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)

By exercising the Licensed Rights (defined below), You accept and agree to be bound by the terms and conditions of this Creative Commons Attribution-NonCommercial 4.0 International Public License ("Public License"). To the extent this Public License may be interpreted as a contract, You are granted the Licensed Rights in consideration of Your acceptance of these terms and conditions, and the Licensor grants You such rights in consideration of benefits the Licensor receives from making the Licensed Material available under these terms and conditions.

**Section 1 – Definitions.**

1. Adapted Material means material subject to Copyright and Similar Rights that is derived from or based upon the Licensed Material and in which the Licensed Material is translated, altered, arranged, transformed, or otherwise modified in a manner requiring permission under the Copyright and Similar Rights held by the Licensor. For purposes of this Public License, where the Licensed Material is a musical work, performance, or sound recording, Adapted Material is always produced where the Licensed Material is synched in timed relation with a moving image.  
2. Adapter's License means the license You apply to Your Copyright and Similar Rights in Your contributions to Adapted Material in accordance with the terms and conditions of this Public License.  
3. Copyright and Similar Rights means copyright and/or similar rights closely related to copyright including, without limitation, performance, broadcast, sound recording, and Sui Generis Database Rights, without regard to how the rights are labeled or categorized. For purposes of this Public License, the rights specified in Section [2(b)(1)-(2)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s2b) are not Copyright and Similar Rights.  
4. Effective Technological Measures means those measures that, in the absence of proper authority, may not be circumvented under laws fulfilling obligations under Article 11 of the WIPO Copyright Treaty adopted on December 20, 1996, and/or similar international agreements.  
5. Exceptions and Limitations means fair use, fair dealing, and/or any other exception or limitation to Copyright and Similar Rights that applies to Your use of the Licensed Material.  
6. Licensed Material means the artistic or literary work, database, or other material to which the Licensor applied this Public License.  
7. Licensed Rights means the rights granted to You subject to the terms and conditions of this Public License, which are limited to all Copyright and Similar Rights that apply to Your use of the Licensed Material and that the Licensor has authority to license.  
8. Licensor means the individual(s) or entity(ies) granting rights under this Public License.  
9. NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation. For purposes of this Public License, the exchange of the Licensed Material for other material subject to Copyright and Similar Rights by digital file-sharing or similar means is NonCommercial provided there is no payment of monetary compensation in connection with the exchange.  
10. Share means to provide material to the public by any means or process that requires permission under the Licensed Rights, such as reproduction, public display, public performance, distribution, dissemination, communication, or importation, and to make material available to the public including in ways that members of the public may access the material from a place and at a time individually chosen by them.  
11. Sui Generis Database Rights means rights other than copyright resulting from Directive 96/9/EC of the European Parliament and of the Council of 11 March 1996 on the legal protection of databases, as amended and/or succeeded, as well as other essentially equivalent rights anywhere in the world.  
12. You means the individual or entity exercising the Licensed Rights under this Public License. **Your** has a corresponding meaning.

**Section 2 – Scope.**

1. **License grant** .  
   1. Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to:  
      1. reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and  
      2. produce, reproduce, and Share Adapted Material for NonCommercial purposes only.  
   2. **Exceptions and Limitations** . For the avoidance of doubt, where Exceptions and Limitations apply to Your use, this Public License does not apply, and You do not need to comply with its terms and conditions.  
   3. **Term** . The term of this Public License is specified in Section [6(a)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s6a) .  
   4. **Media and formats; technical modifications allowed** . The Licensor authorizes You to exercise the Licensed Rights in all media and formats whether now known or hereafter created, and to make technical modifications necessary to do so. The Licensor waives and/or agrees not to assert any right or authority to forbid You from making technical modifications necessary to exercise the Licensed Rights, including technical modifications necessary to circumvent Effective Technological Measures. For purposes of this Public License, simply making modifications authorized by this Section [2(a)(4)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s2a4) never produces Adapted Material.  
   5. Downstream recipients .  
      1. Offer from the Licensor – Licensed Material . Every recipient of the Licensed Material automatically receives an offer from the Licensor to exercise the Licensed Rights under the terms and conditions of this Public License.  
      2. No downstream restrictions . You may not offer or impose any additional or different terms or conditions on, or apply any Effective Technological Measures to, the Licensed Material if doing so restricts exercise of the Licensed Rights by any recipient of the Licensed Material.  
   6. No endorsement . Nothing in this Public License constitutes or may be construed as permission to assert or imply that You are, or that Your use of the Licensed Material is, connected with, or sponsored, endorsed, or granted official status by, the Licensor or others designated to receive attribution as provided in Section [3(a)(1)(A)(i)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s3a1Ai) .  
2. **Other rights** .  
   1. Moral rights, such as the right of integrity, are not licensed under this Public License, nor are publicity, privacy, and/or other similar personality rights; however, to the extent possible, the Licensor waives and/or agrees not to assert any such rights held by the Licensor to the limited extent necessary to allow You to exercise the Licensed Rights, but not otherwise.  
   2. Patent and trademark rights are not licensed under this Public License.  
   3. To the extent possible, the Licensor waives any right to collect royalties from You for the exercise of the Licensed Rights, whether directly or through a collecting society under any voluntary or waivable statutory or compulsory licensing scheme. In all other cases the Licensor expressly reserves any right to collect such royalties, including when the Licensed Material is used other than for NonCommercial purposes.

**Section 3 – License Conditions.**

Your exercise of the Licensed Rights is expressly made subject to the following conditions.

1. **Attribution** .  
   1. If You Share the Licensed Material (including in modified form), You must:  
      1. retain the following if it is supplied by the Licensor with the Licensed Material:  
         1. identification of the creator(s) of the Licensed Material and any others designated to receive attribution, in any reasonable manner requested by the Licensor (including by pseudonym if designated);  
         2. a copyright notice;  
         3. a notice that refers to this Public License;  
         4. a notice that refers to the disclaimer of warranties;  
         5. a URI or hyperlink to the Licensed Material to the extent reasonably practicable;  
      2. indicate if You modified the Licensed Material and retain an indication of any previous modifications; and  
      3. indicate the Licensed Material is licensed under this Public License, and include the text of, or the URI or hyperlink to, this Public License.  
   2. You may satisfy the conditions in Section [3(a)(1)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s3a1) in any reasonable manner based on the medium, means, and context in which You Share the Licensed Material. For example, it may be reasonable to satisfy the conditions by providing a URI or hyperlink to a resource that includes the required information.  
   3. If requested by the Licensor, You must remove any of the information required by Section [3(a)(1)(A)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s3a1A) to the extent reasonably practicable.  
   4. If You Share Adapted Material You produce, the Adapter's License You apply must not prevent recipients of the Adapted Material from complying with this Public License.

**Section 4 – Sui Generis Database Rights.**

Where the Licensed Rights include Sui Generis Database Rights that apply to Your use of the Licensed Material:

1. for the avoidance of doubt, Section [2(a)(1)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s2a1) grants You the right to extract, reuse, reproduce, and Share all or a substantial portion of the contents of the database for NonCommercial purposes only;  
2. if You include all or a substantial portion of the database contents in a database in which You have Sui Generis Database Rights, then the database in which You have Sui Generis Database Rights (but not its individual contents) is Adapted Material; and  
3. You must comply with the conditions in Section [3(a)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s3a) if You Share all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section [4](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s4) supplements and does not replace Your obligations under this Public License where the Licensed Rights include other Copyright and Similar Rights.

**Section 5 – Disclaimer of Warranties and Limitation of Liability.**

1. **Unless otherwise separately undertaken by the Licensor, to the extent possible, the Licensor offers the Licensed Material as-is and as-available, and makes no representations or warranties of any kind concerning the Licensed Material, whether express, implied, statutory, or other. This includes, without limitation, warranties of title, merchantability, fitness for a particular purpose, non-infringement, absence of latent or other defects, accuracy, or the presence or absence of errors, whether or not known or discoverable. Where disclaimers of warranties are not allowed in full or in part, this disclaimer may not apply to You.**  
2. **To the extent possible, in no event will the Licensor be liable to You on any legal theory (including, without limitation, negligence) or otherwise for any direct, special, indirect, incidental, consequential, punitive, exemplary, or other losses, costs, expenses, or damages arising out of this Public License or use of the Licensed Material, even if the Licensor has been advised of the possibility of such losses, costs, expenses, or damages. Where a limitation of liability is not allowed in full or in part, this limitation may not apply to You.**  
3. The disclaimer of warranties and limitation of liability provided above shall be interpreted in a manner that, to the extent possible, most closely approximates an absolute disclaimer and waiver of all liability.

**Section 6 – Term and Termination.**

1. This Public License applies for the term of the Copyright and Similar Rights licensed here. However, if You fail to comply with this Public License, then Your rights under this Public License terminate automatically.  
2. Where Your right to use the Licensed Material has terminated under Section 6(a), it reinstates:  
   1. automatically as of the date the violation is cured, provided it is cured within 30 days of Your discovery of the violation; or  
   2. upon express reinstatement by the Licensor.

For the avoidance of doubt, this Section [6(b)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s6b) does not affect any right the Licensor may have to seek remedies for Your violations of this Public License.

3. For the avoidance of doubt, the Licensor may also offer the Licensed Material under separate terms or conditions or stop distributing the Licensed Material at any time; however, doing so will not terminate this Public License.  
4. Sections [1](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s1) , [5](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s5) , [6](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s6) , [7](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s7) , and [8](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en#s8) survive termination of this Public License.

**Section 7 – Other Terms and Conditions.**

1. The Licensor shall not be bound by any additional or different terms or conditions communicated by You unless expressly agreed.  
2. Any arrangements, understandings, or agreements regarding the Licensed Material not stated herein are separate from and independent of the terms and conditions of this Public License.

**Section 8 – Interpretation.**

1. For the avoidance of doubt, this Public License does not, and shall not be interpreted to, reduce, limit, restrict, or impose conditions on any use of the Licensed Material that could lawfully be made without permission under this Public License.  
2. To the extent possible, if any provision of this Public License is deemed unenforceable, it shall be automatically reformed to the minimum extent necessary to make it enforceable. If the provision cannot be reformed, it shall be severed from this Public License without affecting the enforceability of the remaining terms and conditions.  
3. No term or condition of this Public License will be waived and no failure to comply consented to unless expressly agreed to by the Licensor.  
4. Nothing in this Public License constitutes or may be interpreted as a limitation upon, or waiver of, any privileges and immunities that apply to the Licensor or You, including from the legal processes of any jurisdiction or authority.