import os

path = 'C:/Users/valentyne/Documents/test/'
list_of_files = os.listdir(path)
length = len(list_of_files)

while(True):
    list_of_files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in list_of_files]

    if (len(list_of_files) > 0) and len(list_of_files) > length:
        latest_file = max(paths, key=os.path.getctime)
        print os.path.basename(latest_file)
        length = len(list_of_files)