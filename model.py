with open('data/test.csv') as f:
    test_data = [ int(l.split(',')) for l in f.read().splitlines()[1:] ]

with open('data/test.csv') as f:
    train_data = [ int(l.split(',')) for l in f.read().splitlines()[1:] ]

# TODO