import numpy as np


def init_population(dimension):
    # Inicjalizacja osobnika do ES 1+1
    # np.random.seed(10)
    x = np.random.uniform(-100, 100, size=dimension)
    return x


def evaluate(x, function):
    # Ocena osobnika w ES 1+1
    pass


def get_mutant(x, sigma, dimension):
    y = x + sigma * np.random.normal(loc=0.0, scale=1.0, size=dimension)
    return y

def es(dim, k, init_sigma, max_epoch, function):
    x = init_population(dim)
    succes_mem = []
    sigma = init_sigma   

    epochs = 0
    while(epochs < max_epoch):
        y = get_mutant(x, sigma, dim)
        
        score_x = evaluate(x, function)
        score_y = evaluate(y, function)
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
        epochs += 1



def main():
    # dimension = 50
    # x = init_population(dimension)
    # y = get_mutant(x, dimension)

    a = [True, False, True, True]
    print(sum)




if __name__ == "__main__":
    main()