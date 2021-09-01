from lib import *
from mnist_data import train_data_x, train_data_y, test_data_x, test_data_y

mnist = Model(input_shape=(28*28,))

mnist.add(Layer(512, 28*28))
mnist.add(Layer(10, 512))

def argmax(l):
    return l.index(max(l))

def test():
    TOTAL_RIGHT = 0
    AVERAGE = 0
    for i, t_data in enumerate(test_data_x):
        prediction = mnist.forwardpropagate(t_data)
        if  test_data_y[i][argmax(prediction)] == 1:
            TOTAL_RIGHT += 1
        
        if AVERAGE == 0:
            AVERAGE = mnist.get_error(t_data, test_data_y[i])
        else:
            AVERAGE = (AVERAGE * i + mnist.get_error(t_data, test_data_y[i])) / (i + 1)

        #if i % 1000 == 0:
        #    print(f"{TOTAL_RIGHT}/{i}")
        #    print(f"Average Cost: {AVERAGE}")

    print(f"{TOTAL_RIGHT}/{i} = {TOTAL_RIGHT / i}")
    print(f"Average Cost: {AVERAGE}")

test()

for k in range(30):
    print(f"Epoch {k}")
    mnist.train_batch(train_data_x, train_data_y, learning_rate=0.5)
    test()
    print("-"*10)


