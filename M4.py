import os
import shutil

src_dir = os.getcwd() #get the current working dir
print(src_dir)

# create a dir where we want to copy and rename
os.listdir()
id=1
dest_dir = src_dir+"/user_files"
src_file = os.path.join(src_dir, 'trusted.csv')
shutil.copy(src_file,dest_dir) #copy the file to destination dir

dst_file = os.path.join(dest_dir,'trusted.csv')
new_dst_file_name = os.path.join(dest_dir, 'trusted'+str(id)+'.csv')

os.rename(dst_file, new_dst_file_name)#rename
os.chdir(dest_dir)

print(os.listdir())