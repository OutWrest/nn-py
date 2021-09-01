from lib import *

def mnist():
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

def logic_xor():
    xor = Model(input_shape=(2,))
    xor.add(Layer(2, 2))
    xor.add(Layer(1, 2))

    x, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]

    def test():
        TOTAL_RIGHT = 0
        AVERAGE = 0

        for i, x_data in enumerate(x):
            prediction = xor.forwardpropagate(x_data)
            print(prediction, y[i][0])
            if  y[i][0] == round(prediction[0]):
                TOTAL_RIGHT += 1
            
            if AVERAGE == 0:
                AVERAGE = xor.get_error(x_data, y[i])
            else:
                AVERAGE = (AVERAGE * i + xor.get_error(x_data, y[i])) / (i + 1)

        print(f"{TOTAL_RIGHT}/{i + 1} = {TOTAL_RIGHT / (i + 1)}")
        print(f"Average Cost: {AVERAGE}")

    for epoch in range(100000 + 1):
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}")

        for tx, ty in zip(x, y):
            xor.train(tx, ty, learning_rate=0.5)

    test()

def logic_and():
    and_ = Model(input_shape=(2,))
    and_.add(Layer(2, 2))
    and_.add(Layer(1, 2))

    x, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [0], [0], [1]]

    def test():
        TOTAL_RIGHT = 0
        AVERAGE = 0

        for i, x_data in enumerate(x):
            prediction = and_.forwardpropagate(x_data)
            print(prediction, y[i][0])
            if  y[i][0] == round(prediction[0]):
                TOTAL_RIGHT += 1
            
            if AVERAGE == 0:
                AVERAGE = and_.get_error(x_data, y[i])
            else:
                AVERAGE = (AVERAGE * i + and_.get_error(x_data, y[i])) / (i + 1)

        print(f"{TOTAL_RIGHT}/{i + 1} = {TOTAL_RIGHT / (i + 1)}")
        print(f"Average Cost: {AVERAGE}")

    for epoch in range(10000 + 1):
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}")
        
        for tx, ty in zip(x, y):
            and_.train(tx, ty, learning_rate=0.5)

    test()

def logic_or():
    or_ = Model(input_shape=(2,))
    or_.add(Layer(2, 2))
    or_.add(Layer(1, 2))

    x, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]]

    def test():
        TOTAL_RIGHT = 0
        AVERAGE = 0

        for i, x_data in enumerate(x):
            prediction = or_.forwardpropagate(x_data)
            print(prediction, y[i][0])
            if  y[i][0] == round(prediction[0]):
                TOTAL_RIGHT += 1
            
            if AVERAGE == 0:
                AVERAGE = or_.get_error(x_data, y[i])
            else:
                AVERAGE = (AVERAGE * i + or_.get_error(x_data, y[i])) / (i + 1)

        print(f"{TOTAL_RIGHT}/{i + 1} = {TOTAL_RIGHT / (i + 1)}")
        print(f"Average Cost: {AVERAGE}")

    for epoch in range(10000 + 1):
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}")
        
        for tx, ty in zip(x, y):
            or_.train(tx, ty, learning_rate=0.5)

    test()


if __name__ == "__main__":
    logic_or()

