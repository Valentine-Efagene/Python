#-*-coding:utf8;-*-
#qpy:3
#qpy:console

def climbingLeaderboard(scores, alice):
    result = []
    a = scores[:]
    a = list(set(a))
    a = sorted(a)[::-1]
    i = len(a) - 1

    for x in alice:
        while i >= 0:
            if x >= a[i]:
                i -= 1
            else:
                result.append(i + 2)
                break
        if i < 0:
            result.append(1)
    
    print(result)

scores = [100, 100, 50, 40, 40, 20, 10]
alice = [5, 25, 50, 120]
climbingLeaderboard(scores, alice) 