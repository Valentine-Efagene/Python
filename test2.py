with open('demofile.txt') as fp:
    lines = fp.read().split("\n")

for l in lines:
    print l.replace('\\', '/')