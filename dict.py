#/usr/bin/env python
# -*- coding:utf-8 -*-
'''
	外部传入一个参数：文件名/文件目录，读取其中的文本文件，建立字典，以空格+tab+换行作为单词的分割，英文不区分大小写
'''
import sys,os
import glob
def build_vocab(inpath,dest_path):
	vocab=set()
	for f in inpath:
		fin = open(f,'r')
		for line in fin:
			line = line.replace('\n','')
			line = line.replace('\r','')
			line = line.lower()
			vocab |=set(line.split())
	fout = open(dest_path,'w')
	for w in sorted(vocab):
		fout.write(w+'\n')

def get_all_files(path):
	path_box = []
	if not os.path.isdir(path):	#如果当前文件不是目录，则直接返回
		path_box.append(path)
		return path_box
	# 当前是一个目录
	for item in os.listdir(path):
		if path.endswith('/'):
			path_box += get_all_files(path+item)
		else:
			path_box += get_all_files(path+'/'+item)
	return path_box

if __name__ =='__main__':
	#format: infile/dir  outfile
	inf = sys.argv[1]
	outf = sys.argv[2]
	infs = get_all_files(inf)
	build_vocab(infs,outf)
