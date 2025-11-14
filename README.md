#  COMIX: Confidence-based logit-level graph mixup

This project is a PyTorch implementation of **COMIX**  
(*Confidence-based logit-level graph mixup*, PLOS ONE 2025).  
COMIX proposes a confidence-guided sampling and adaptive logit-level mixup framework that achieves  
accurate graph classification under multi-positive unlabeled (MPU) learning settings.

---

## Prerequisites

Our implementation is based on Python 3.9 and PyTorch Geometric.  
Please refer to `requirements.txt` for the complete package list.

- Python ≥ 3.9  
- PyTorch ≥ 1.12.0  
- PyTorch Geometric ≥ 2.0.0  

---

## Datasets

COMIX uses five widely-used citation and co-purchase network datasets,  
automatically downloaded and preprocessed via PyTorch Geometric.  

Each dataset is converted into the **Multi-Positive Unlabeled (MPU)** format,  
where a subset of positive classes is labeled while others remain unlabeled.

| Dataset | Nodes | Edges | Features | P1 | P2 | P3 | Negatives |
|:---------|------:|------:|----------:|---:|---:|---:|----------:|
| Computers | 13,752 | 491,722 | 767 | 5,158 | 2,156 | 2,142 | 4,296 |
| Cora-ML | 2,995 | 16,316 | 2,879 | 857 | 452 | 442 | 1,244 |
| CiteSeer | 3,327 | 9,104 | 3,703 | 701 | 668 | 596 | 1,362 |
| CiteSeer-full | 4,230 | 10,674 | 602 | 831 | 778 | 740 | 1,881 |
| DBLP | 17,716 | 105,734 | 1,639 | 7,920 | 5,645 | 2,169 | 1,982 |

---

## Usage

To reproduce the experimental results from the paper, simply run:

```bash
bash demo.sh
```

The script automatically:
1. Downloads and preprocesses datasets  
2. Initializes the COMIX GCN backbone  
3. Applies confidence-based soft sampling and mixup  
4. Trains and evaluates the model

You can modify experimental arguments such as:
- `--epochs`, `--trn-ratio` in `main.py`
- model hyperparameters (`--units`, `--layers`)  
- optimizer and loss settings in `train.py`

Example manual execution:
```bash
python src/main.py --data CiteSeer --epochs 500 --trn-ratio 0.2 --gpu 0
```

---

## 🧠 Method Overview

COMIX integrates two major mechanisms:

1. **Confidence-based Soft Sampling**  
   - Softly sample partners to augment nodes proportionally to their predicted confidence scores  

2. **Confidence-based Logit-Mixing**  
   - Mixes logits between nodes and soft partners  
   - Adjusts mixing ratio λ based on confidence to prevent contamination  

Together, these modules enable **reliable negative discovery** and **stable multi-class PU optimization**.

---

## Citation

Please cite the following paper if you use our code:

```
@article{yoon2025cosmos,
  title   = {COMIX: Confidence-based logit-level graph mixup},
  author  = {Yoon, Hoyoung and Kim, Junghun and Park, Shihyung and Kang, U},
  journal = {PLOS ONE},
  year    = {2025},
  publisher = {Public Library of Science},
  doi     = {10.1371/journal.pone.xxxxxxx}
}
```

---

## License

This software may be used **only for research evaluation purposes**.  
For other purposes (e.g., commercial use), please contact the authors.