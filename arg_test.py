import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='Rounak')
    parser.add_argument('-a', '--age', type=float, default=22)
    args = parser.parse_args()

    print(args.name, args.age)