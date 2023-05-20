import numpy as np
from cec2017.functions import f3 as func

def init_population(dimension, samples):
    # Inicjalizacja osobnika do ES 1+1
    # np.random.seed(10)
    x = np.random.uniform(-50, 50, size=(samples, dimension))
    return x


def evaluate(x):
    val = func(x)
    return val


def get_mutant(x, sigma, dimension):
    y = x + sigma * np.random.normal(loc=0.0, scale=1.0, size=dimension)
    return y


def es(dim, k, init_sigma, max_epoch):
    x = init_population(dim,samples=1)
    succes_mem = []
    sigma = init_sigma
    score_x = None
    score_y = None

    epochs = 0
    while(epochs < max_epoch):
        y = get_mutant(x, sigma, dim)
        
        score_x = evaluate(x)
        score_y = evaluate(y)
        if score_y < score_x:
            succes_mem.append(True)
            x = y
        else:
            succes_mem.append(False)

        if epochs%k == 0:
            if sum(succes_mem)/len(succes_mem) >= 0.2:
                sigma = 1.22*sigma
            elif sum(succes_mem)/len(succes_mem) < 0.2:
                sigma = 0.82*sigma
            succes_mem = []
        print((epochs, score_x[0]))
        epochs += 1


    return (x, score_x)

def main():
    print(func([[0, 0]]))
    # print(func)
    # a = es(dim=2, k=10, init_sigma=1, max_epoch=10000)
    # print(a)




if __name__ == "__main__":
    main()