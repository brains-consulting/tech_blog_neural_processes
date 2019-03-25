# tech blog neural processes
An implementation of the Neural Processes
tech blog: https://blog.brains-consulting.tech/

## Usage
install the package
```bash
pip install git+https://github.com/brains-consulting/tech_blog_neural_processes
```

## run examples

clone the repository
```bash
git clone https://github.com/brains-consulting/tech_blog_neural_processes
cd examples
```

### Toy dataset
run the toy dataset sample (imitated deepmind's implementation)
```bash
python train_toy.py --epochs=20000 --log-interval=1000
```

### MNIST
run the mnist dataset sample
```bash
python train_toy.py --lr=0.001 --batch-size=30 --epochs=500 --log-interval=100 --fix-iter=100 --dataset="mnist"
```

### Fashion-MNIST
run the fashion-mnist dataset sample
```bash
python train_toy.py --lr=0.001 --batch-size=30 --epochs=500 --log-interval=100 --fix-iter=100 --dataset="fashion"
```

