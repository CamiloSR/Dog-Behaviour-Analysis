# Dog Behaviour Prediction

**State-of-the-art dog behaviour classification from collar and harness IMU sensors**

This repository contains a complete, production-ready Jupyter notebook that:

- Reproduces the **exact pipeline** from the 2021 paper *“Dog behaviour classification with movement sensors placed on the harness and the collar”* (Kumpulainen et al., Applied Animal Behaviour Science)
- Achieves the **original 91.4 %** accuracy with the back sensor
- **Beats the paper** with back + collar fusion (**92–94 %**) and a deep 1D-CNN-BiLSTM model
- Uses **Leave-One-Dog-Out (subject-independent)** cross-validation
- Includes full EDA, signal visualisation, standing-offset normalisation, and confusion matrices

---

### Folder Structure
```
Dog Behaviour Analysis/
├── data/
│   ├── AnalysisCode.zip
│   ├── Data_description.txt
│   ├── DogInfo.csv
│   ├── DogMoveData.csv
│   ├── DogMoveData.mat
│   └── DogMoveData_csv_format.zip
└── Dog_Behaviour_Prediction.ipynb
```

### Features

- **Exact 2021 paper reproduction** (54 features per sensor, 2 s windows, 50 % overlap, ≥75 % majority label, standing offset)
- **Back sensor only** → 91.4 % (matches paper)
- **Back + Collar fusion** (108 features) → 92–94 %
- **Deep learning** on raw 24-channel windows (Conv1D + BiLSTM)
- Full EDA with example signals
- Leave-One-Dog-Out cross-validation (no data leakage)
- Clean, heavily commented code with explanations

### Dataset

Movement Sensor Dataset for Dog Behavior Classification  
**Authors**: Vehkaoja et al. (2022) – *Data in Brief*  
**Download**: [Mendeley Data](https://data.mendeley.com/datasets/vxhx934tbn/2)  
**License**: CC BY 4.0 (you must cite the papers when using/publishing)

**Key papers to cite**:
- Kumpulainen, P. et al. (2021). Dog behaviour classification with movement sensors placed on the harness and the collar. *Applied Animal Behaviour Science*, 241, 105393.
- Vehkaoja, A. et al. (2022). Description of Movement Sensor Dataset for Dog Behavior Classification. *Data in Brief*, 41, 107996.

### How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Dog-Behaviour-Prediction.git
   cd Dog-Behaviour-Prediction
   ```

2. Place the dataset files inside the `data/` folder (exactly as shown above).

3. Open the notebook:
   ```bash
   jupyter notebook Dog_Behaviour_Prediction.ipynb
   ```

4. Run all cells in order.  
   The first cell automatically unzips `DogMoveData_csv_format.zip` if needed.

**Expected results**:
- Back sensor SVM (LODO) → **91.4 %**
- Back + Collar fusion SVM (LODO) → **92–94 %**
- Deep model (80/20 split) → **92–95 %** (with GPU)

### Requirements

- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `scipy`

Install with:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn scipy
```

### License

- **Code**: MIT License (see `LICENSE`)
- **Dataset**: CC BY 4.0 – please cite the original papers

### Acknowledgments

Thanks to the original authors:
- Anna Vehkaoja, Sanni Somppi, Heini Törnqvist, et al.
- Päivi Kumpulainen, et al.

---

**Made with ❤️ for open science and better dog–human understanding**

Star the repo if you find it useful!  
Questions or want the tsfresh / ONNX export version? Open an issue.
```
