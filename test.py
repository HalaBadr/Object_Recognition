from Model import *

def main():
    print('\nCreating Model....')
    model = ObjectsModel(100)

    print('Training...')
    if not model.trained :
        train_fit = model.fit()

    print('\nTesting...')
    result = model.test()
    print("Result:", result)

if __name__ == '__main__': main()