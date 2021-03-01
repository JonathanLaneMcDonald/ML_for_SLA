
import os
'''
for f in sorted([x for x in open('xz_files','r').read().split('\n') if len(x)]):
	print ('extracting',f)
	os.system('unxz -c '+f+' > '+f.split('/')[-1].split('.')[0])
	print ('processing '+f.split('/')[-1].split('.')[0])
	os.system('python simple_comment_filter.py '+f.split('/')[-1].split('.')[0])
	print ('removing '+f.split('/')[-1].split('.')[0])
	os.system('rm '+f.split('/')[-1].split('.')[0])
'''
for f in sorted([x for x in open('zst_files','r').read().split('\n') if len(x)]):
	print ('extracting',f)
	os.system('unzstd '+f+' -o '+f.split('/')[-1].split('.')[0])
	print ('processing '+f.split('/')[-1].split('.')[0])
	os.system('python simple_comment_filter.py '+f.split('/')[-1].split('.')[0])
	print ('removing '+f.split('/')[-1].split('.')[0])
	os.system('rm '+f.split('/')[-1].split('.')[0])


