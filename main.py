from lib import *
from image_data import train_data, test_data

mnist = Model(input_shape=(28*28,))

mnist.add(Layer(64, 28*28))
mnist.add(Layer(10, 64))

def test():
    p = [0 for _ in range(10)]
    p[train_data[0][0]] = 1
    print(f"Total Error: {mnist.get_error(train_data[0][1:], p)}")

for i, t_data in enumerate(train_data):
    p = [0 for _ in range(10)]
    p[t_data[0]] = 1

    mnist.train(t_data[1:], p, learning_rate=0.1)

    if i % 1000 == 0:
        print(f"Epoch {i}")
        test()

def argmax(l):
    return l.index(max(l))

with open('out.csv', 'w') as f:
    f.write("ImageId,Label")
    for i, t_data in enumerate(test_data):
        if i % 1000 == 0:
            print(f"Test {i}")

        prediction = mnist.forwardpropagate(t_data)
        print(prediction)

        f.write(f"{i+1},{argmax(prediction)}\n")