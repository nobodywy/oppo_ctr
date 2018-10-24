def read_row_data():
    f = open('./data/vail.txt',encoding='utf-8')
    tag_s = set()
    tag_dict = {}
    a = [0,0]
    for line in f:
        l = line.split('\t')
        label = int(l[4])
        if(label):
            a[0] += 1
        else:
            a[1] += 1
        if(l[3] in tag_dict):
            if(label):
                tag_dict[l[3]][0] += 1
            else:
                tag_dict[l[3]][1] += 1
        else:
            if(label):
                tag_dict[l[3]] = [1,0]
            else:
                tag_dict[l[3]] = [0,1]
    print(tag_dict)
    print(a)
if __name__ == '__main__':
    read_row_data()