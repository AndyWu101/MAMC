# Multi-Actor Multi-Critic Deep Deterministic Reinforcement Learning with a Novel Q-Ensemble Method


## Paper

[https://arxiv.org/abs/2510.01083](https://arxiv.org/abs/2510.01083)


## Requirements

Step 1: Install CUDA 12.6

Step 2: Install Python 3.12.7

Step 3: Install dependencies:
```setup
pip install -r requirements.txt
```

Step 4: Install PyTorch
```setup
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

## Training

To train the model(s) in the paper,

run this command:

```setup
python main.py --save_result --env_name "HalfCheetah-v5" --seed 1000
```
You will get a folder named `Result`, which contains a JSON file with the experiment results, e.g., `[MAMC][HalfCheetah-v5][1000][2025-05-15][Learning Curve][XGHPJR].json`

To obtain multiple experiment results, run the following bash.

```bash
for ((i = 1000; i < 1010; i += 1))
do
    python main.py \
        --save_result \
        --env_name "HalfCheetah-v5" \
        --seed $i
done
```

## Results

![](assets/[Main%20Result][TD3-SMR,%20DARC-SMR,%20MAMC].png)
![](assets/[Main%20Result][SAC-SMR,%20REDQ-SMR,%20MAMC].png)


## License

[LICENSE](LICENSE)