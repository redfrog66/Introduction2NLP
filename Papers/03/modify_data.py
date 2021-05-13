#!/usr/bin/env python3


def main():
    with open('tripadvisor_hotel_reviews.csv', 'r') as f:
        data = f.read().strip()
    #
    data = [item for item in data.split('\n')]
    header = data[0]

    for i in range(1, len(data)):
        data[i] = data[i].split('",')

    for i in range(1, len(data)):
        data[i][0] = data[i][0] + '"'

    for i in range(1, len(data)):
        if int(data[i][1]) == 1:
            data[i].append('worst')
        elif int(data[i][1]) == 2:
            data[i].append('bad')
        elif int(data[i][1]) == 3:
            data[i].append('ok')
        elif int(data[i][1]) == 4:
            data[i].append('good')
        elif int(data[i][1]) == 5:
            data[i].append('best')
    #

    for i in range(1, len(data)):
        tmp = int(data[i][1])
        tmp -= 1
        data[i][1] = str(tmp)
        

    with open('tripadvisor_hotel_reviews_modified.csv', 'w') as f1:
        print(header, file=f1)

        for i in range(1, len(data)):
            for j in range(len(data[i])):
                if j == len(data[i])-1:
                    print(data[i][j], file=f1, end='')
                else:
                    print(data[i][j], file=f1, end='')
                    print(',', file=f1, end='')
            print('', file=f1)
    #

    #print(data)

#################

if __name__ == "__main__":
    main()