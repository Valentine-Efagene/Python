import itertools
import subprocess

template = "abcdefghijklmnopqrstuvwxyz1234567890#$_+-?\'\"\/ABCDEFGHIJKLMNOPQRSTUVWXYZ"
print("Running...")

for charLength in range(3, 10):
    passwords = itertools.product(template, repeat=charLength)

    for i in passwords:
        password = ''.join(i)
        print(password)
        password = "-p" + password
        error = subprocess.call( ['unrar', 'x', password, 'C:\\Users\\valentyne\\Desktop\\test.rar'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )

        if (error == 0):
            print('password = ', ''.join(i))
            exit(0)
        else:
            #os.rmdir("C:\\Users\\valentyne\\Desktop\\test") # If folder
            continue