import os
import sys
offset = int(sys.argv[1]) if(len(sys.argv) > 1 and sys.argv[1].isdigit()) else 0
file_name_ext = sys.argv[2] if len(sys.argv) > 2 else 'case-'
path = sys.argv[3] if len(sys.argv) > 3 else '.'
files = sorted(os.listdir(path), reverse= True, key= lambda file : int(os.path.splitext(file)[0]) if (os.path.splitext(file)[0].isdigit()) else -1)
for file in files:
    file_name, ext = os.path.splitext(file)
    if(file_name.isdigit()):
        
        os.rename(os.path.join(path,file),os.path.join(path,file_name_ext + str(int(file_name) + offset) + ext))

