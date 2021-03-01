with open('data/test.csv', 'r') as f:
    test_data = [ [ int(px) for px in l.split(',') ] for l in f.read().splitlines()[1:] ]
    
with open('data/train.csv', 'r') as f:
    train_data = [ [ int(px) for px in l.split(',') ] for l in f.read().splitlines()[1:] ]