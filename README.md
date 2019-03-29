# tech blog neural processes
An implementation of the Neural Processes
on our tech blog: https://blog.brains-consulting.tech/entry/2019/03/29/180000

See Also:
- https://blog.brains-consulting.tech/entry/2019/03/04/120000
- https://blog.brains-consulting.tech/

## Usage
install the package
```bash
pip install git+https://github.com/brains-consulting/tech_blog_neural_processes
```

## run examples

### preparation
clone the repository
```bash
git clone https://github.com/brains-consulting/tech_blog_neural_processes
```

### Toy dataset
run the toy dataset example (imitated deepmind's implementation)
```bash
cd $(git rev-parse --show-toplevel)/examples
python train_toy.py --epochs=20000 --log-interval=1000
```

### MNIST
run the mnist dataset example
```bash
cd $(git rev-parse --show-toplevel)/examples
python train_mnists.py --lr=0.001 --batch-size=100 --epochs=300 --log-interval=100 --fix-iter=100 --seed=123 --dataset="mnist"
```

### Fashion-MNIST
run the fashion-mnist dataset example
```bash
cd $(git rev-parse --show-toplevel)/examples
python train_mnists.py --lr=0.001 --batch-size=100 --epochs=300 --log-interval=100 --fix-iter=100 --seed=123 --dataset="fashion"
```

### Kuzushiji-MNIST
run the fashion-mnist dataset example
```bash
cd $(git rev-parse --show-toplevel)/examples
python train_mnists.py --lr=0.001 --batch-size=100 --epochs=300 --log-interval=100 --fix-iter=100 --seed=123 --dataset="kuzushiji"
```

## run visdom server
you may run the visdom server on another terminal before you ran the examples like above
```bash
sh $(git rev-parse --show-toplevel)/bin/visdom.sh
```
