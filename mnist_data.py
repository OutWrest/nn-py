with open('data/mnist_test.csv', 'r') as f:
    data = f.read().splitlines()
    test_data_x = [ [ int(px) / 255 for px in l.split(',') ][1:] for l in data[1:] ]
    test_data_y = []
    for l in data[1:]:
        p = [0 for _ in range(10)]
        p[int(l[0])] = 1
        test_data_y.append(p)
    
with open('data/mnist_train.csv', 'r') as f:
    data = f.read().splitlines()
    train_data_x = [ [ int(px) / 255 for px in l.split(',') ][1:] for l in data[1:] ]
    train_data_y = []
    for l in data[1:]:
        p = [0 for _ in range(10)]
        p[int(l[0])] = 1
        train_data_y.append(p)