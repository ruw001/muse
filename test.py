import random
def randomTaskGenerator(N, length):
    '''
    make sure at least 10 matches in the sequence
    '''
    print('Generating sequence...')
    seq = [''] * length
    res = [0] * length
    letters = ['B', 'F', 'H', 'J', 'L', 'M', 'Q', 'R', 'X']
    for i in range(10):
        pairhead = random.randint(0, length-1-N)
        while seq[pairhead] != '' or seq[pairhead + N] != '':
            pairhead = random.randint(0, length-1-N)
        num = random.randint(0,8)
        seq[pairhead] = letters[num]
        seq[pairhead + N] = letters[num]
    for i in range(length):
        if seq[i] == '':
            num = random.randint(0,8)
            seq[i] = letters[num]
    for i in range(N, length):
        if seq[i-N] == seq[i]:
            res[i] = 1
    
    print('Sequence generated!')
    return seq, res

# seq, res = randomTaskGenerator(3, 30)
# print(seq)
# print(res)

times = []
mistakes = 0
total = 0

with open('ruru/task_Adaptive01.log', 'r') as infile:
    lines = infile.readlines()
    for l in lines:
        items = l.strip().split(',')
        if len(items) != 4:
            continue
        total += 1
        if items[2] == 'False':
            mistakes +=1
        if items[3] != 'N/A':
            times.append(float(items[3]))
print('Avg time: {}'.format(sum(times)/len(times)))
print('Acc: {}'.format(1 - mistakes/total))
