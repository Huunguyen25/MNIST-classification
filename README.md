# MNIST Classification (Simple Run Guide)

## 1) Install dependencies

```bash
activate env
python3 -m pip install -r requirements.txt
```

## 2) Run the MLP model

```bash
python3 MNIST/MLP_template.py
```

This prints training/evaluation metrics and saves a training-loss graph as `mlp_training_loss.png`.

## 3) Run the CNN model

```bash
python3 MNIST/CNN_template.py
```

This prints training/evaluation metrics and saves a training-loss graph as `cnn_training_loss.png`.
