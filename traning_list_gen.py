import glob
import os

path = input('''This is for X-ray data set. Please copy and paste path to your local data dir:
blank will be set to current directory  ''')
if path in (''):
    path = os.getcwd()
os.chdir(path)
std_list = glob.glob('./*standar*.tif')
dark_list = []
for el in std_list:
    if 'dark' in el:
        dark_list.append(el)
        std_list.remove(el)
print('std_list = {}'.format(std_list))
print('dark_list = {}'.format(dark_list))

