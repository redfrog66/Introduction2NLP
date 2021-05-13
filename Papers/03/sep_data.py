#!/usr/bin/env python3

from random import shuffle


def main():
    with open('tripadvisor_hotel_reviews_modified.csv', 'r') as f:
        data = f.read().strip()
    #

    data = [item for item in data.split('\n')]
    header = data[0]
    data = data[1:]
    shuffle(data)

    with open('tripadvisor_train.csv', 'w') as fo1:
        print(header, file=fo1)
        for i in range(0, 12295):
            print(data[i], file=fo1)
        #
    #

    with open('tripadvisor_val.csv', 'w') as fo2:
        print(header, file=fo2)
        for i in range(12294, 16393):
            print(data[i], file=fo2)
        #
    #

    with open('tripadvisor_test.csv', 'w') as fo3:
        print(header, file=fo3)
        for i in range(16392, len(data)):
            print(data[i], file=fo3)
        #
    #

#################

if __name__ == "__main__":
    main()