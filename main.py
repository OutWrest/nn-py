from lib import *
from image_data import train_data_x, train_data_y, test_data_x

mnist = Model(input_shape=(28*28,))

mnist.add(Layer(128*2, 28*28))
mnist.add(Layer(10, 128*2))

def test():
    print(f"Total Error: {mnist.get_error(train_data_x[0], train_data_y[0])}")

for k in range(5):
    mnist.train_batch(train_data_x, train_data_y, learning_rate=0.25)
    test()

def argmax(l):
    return l.index(max(l))

print("TESTING")

with open('out.csv', 'w') as f:
    f.write("ImageId,Label\n")
    for i, t_data in enumerate(test_data_x):
        if i % 1000 == 0:
            print(f"Test {i}")

        prediction = mnist.forwardpropagate(t_data)
        #print(prediction)

        f.write(f"{i+1},{argmax(prediction)}\n")