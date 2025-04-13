


# EDC²-RAG: Efficient Dynamic Clustering-Based Document Compression for Retrieval-Augmented-Generation

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains the official implementation of **EDC²-RAG**, a plug-and-play document preprocessing framework that enhances Retrieval-Augmented Generation (RAG) by dynamically clustering and compressing retrieved documents. Our method improves the robustness, relevance, and factuality of LLM-based generation systems by leveraging fine-grained inter-document relationships.

> 📄 [Read the Paper on arXiv](https://arxiv.org/abs/2504.03165)  
> 🔬 Developed by [Tsinghua University NLP](https://github.com/thunlp)

## 🔍 Overview

Retrieval-Augmented Generation (RAG) enhances LLM outputs by integrating external documents. However, current RAG systems often suffer from **noise**, **redundancy** in retrieved content.

**EDC²-RAG** addresses these issues via:
- 🔗 **Dynamic Clustering** of documents based on semantic similarity.
- ✂️ **Query-aware Compression** using LLMs to eliminate irrelevant or redundant content.
- 🧠 A more informative and coherent context for generation.

![Overview](pictures/overview.jpg)

## 🚀 Features

- 📚 **Noise & Redundancy Reduction**: Fine-grained document-level structuring.
- 🧩 **Plug-and-Play**: No fine-tuning required, compatible with any retriever or LLM.
- ⚡ **Efficient**: Reduces hallucinations while minimizing inference overhead.
- 🧪 **Extensive Evaluation**: Verified across hallucination detection and QA tasks.

## 🧱 Architecture

1. **Document Retrieval**  
   Standard retriever (e.g., DPR) fetches top-k documents.

2. **Dynamic Clustering**  
   Documents are grouped based on similarity to the query and each other.

3. **LLM-based Compression**  
   Each cluster is summarized using prompts tailored to the query.

4. **Answer Generation**  
   The refined, dense context is passed to the LLM for final answer generation.

## 📊 Experimental Results

| Dataset        | Metric    | RALM | Raw Compression | EDC²-RAG (Ours) |
|----------------|-----------|------|------------------|------------------|
| TriviaQA       | F1 Score  | 93.78 | 93.29           | **93.81**        |
| WebQ           | F1 Score  | 88.75 | 88.25           | **89.23**        |
| HaluEval       | Accuracy  | 76.93 | 77.80           | **78.85**        |
| FELM           | Bal. Acc. | 55.65 | 61.89           | **62.26**        |

See the paper for full ablation studies and robustness testing.

### Our code and datasets will be uploaded before 23:59, April 13 (Anywhere on Earth, AOE).
🕒 Deadline: **[April 13, 2025 23:59 AOE](https://www.timeanddate.com/worldclock/fixedtime.html?msg=Deadline&iso=20250413T2359&p1=1440)**  

## 📄 Citation

If you find this project useful, please consider citing:

```bibtex
@article{li2024efficient,
  title={Efficient Dynamic Clustering-Based Document Compression for Retrieval-Augmented-Generation},
  author={Li, Weitao and Liu, Kaiming and Zhang, Xiangyu and Lei, Xuanyu and Ma, Weizhi and Liu, Yang},
  journal={arXiv preprint arXiv:2504.03165},
  year={2024}
}
```

## 🧠 Contact

For questions or collaborations, please open an issue or contact us at:

- Weitao Li — liwt23@mails.tsinghua.edu.cn
- Kaiming Liu — lkm20@mails.tsinghua.edu.cn

---
