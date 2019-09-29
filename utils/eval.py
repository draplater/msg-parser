import sys


def empty_tree_to_tree(token):
	newtoken = []
	newid = []
	for node in token:
		newid.append(len(newtoken))
		if node[1] != 'EMCAT':
			newtoken.append(node)
		node[2] = int(node[2])
	for node in newtoken:
		if node[2] != -1: node[2] = newid[node[2]]
	return newtoken

def empty_tree_empty_arc(token):
	newtoken = []
	newarc = []
	newid = []
	for node in token:
		node[2] = int(node[2])
		newid.append(len(newtoken))
		if node[1] != 'EMCAT':
			newtoken.append(node)
		else:
			newarc.append([node[2], len(newtoken)])
	for arc in newarc:
		if arc[0] != -1: arc[0] = newid[arc[0]]
	return newarc

def eval_empty_node(file1, file2, name):
	ttp, ttr, ct = 0, 0, 0
	with open(file1, 'r') as f1:
		with open(file2, 'r') as f2:
			while True:
				treeend = False
				token1, token2 = [], []
				while True:
					l1 = f1.readline()
					if not l1:
						treeend = True
						break
					l1 = l1.strip()
					if not l1: break
					token1.append(l1.split('\t'))
				while True:
					l2 = f2.readline()
					if not l2:
						treeend = True
						break
					l2 = l2.strip()
					if not l2: break
					token2.append(l2.split('\t'))
				if treeend: break
				arcs1 = empty_tree_empty_arc(token1)
				arcs2 = empty_tree_empty_arc(token2)
				ttp += len(arcs2)
				ttr += len(arcs1)
				for arc in arcs2:
					for a in arcs1:
						if a[0] == arc[0] and a[1] == arc[1]: ct += 1
	p = float(ct) / float(ttp)
	r = float(ct) / float(ttr)
	f = 2 * p * r / (p + r)
	return '			P:', round(p * 100.0, 2), 'R:', round(r * 100.0, 2), 'F:', round(f * 100.0, 2)

def eval_empty_tree(file1, file2, name):
	tt, ct = 0, 0
	with open(file1, 'r') as f1:
		with open(file2, 'r') as f2:
			while True:
				treeend = False
				token1, token2 = [], []
				while True:
					l1 = f1.readline()
					if l1.startswith("#"):
						continue
					if not l1:
						treeend = True
						break
					l1 = l1.strip()
					if not l1: break
					token1.append(l1.split('\t'))
				while True:
					l2 = f2.readline()
					if l2.startswith("#"):
						continue
					if not l2:
						treeend = True
						break
					l2 = l2.strip()
					if not l2: break
					token2.append(l2.split('\t'))
				if treeend: break
				ntoken1 = empty_tree_to_tree(token1)
				ntoken2 = empty_tree_to_tree(token2)
				if len(ntoken1) != len(ntoken2):
					print('fuck')
					print(ntoken1)
					continue
				for i in range(0, len(ntoken1)):
					if (ntoken1[i][1] != 'PU' and ntoken1[i][1][0].isalpha()):
						tt += 1
						if ntoken1[i][2] == ntoken2[i][2]: ct += 1
	return name, round(float(ct) / float(tt) * 100.0, 2)

# print('10 ITERATION')
# print('--------CN DEV--------')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\2nd.dev.out', '  2nd')
# print('')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\ec2nd.dev.out', 'ec2nd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\ec2nd.dev.out', 'ec2nd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out', 'dd2nd')
# print('------------')

# eval_empty_tree('cn.dev.tree', 'cn_feat\\2nd.dev.out', '   2nd')
# print('')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\ec2ndf.dev.out', 'ec2ndf')
# eval_empty_node('cn.dev.tree', 'cn_feat\\ec2ndf.dev.out', 'ec2ndf')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\comb2ndf.dev.out', 'comb2ndf')
# eval_empty_node('cn.dev.tree', 'cn_feat\\comb2ndf.dev.out', 'comb2ndf')
# eval_empty_tree('cn_feat\\dd2ndf.dev.out', 'cn_feat\\comb2ndf.dev.out', 'similarity')
# eval_empty_node('cn_feat\\dd2ndf.dev.out', 'cn_feat\\comb2ndf.dev.out', 'similarity')
# print('------------')

# eval_empty_tree('cn.dev.tree', 'cn_feat\\3rd.dev.out', '  3rd')
# print('')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\ec3rd.dev.out', 'ec3rd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\ec3rd.dev.out', 'ec3rd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out', 'dd3rd')
# print('------------')

# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ec3rd.dev.out', 'dd2ec3rd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ec3rd.dev.out', 'dd2ec3rd')
# print('------------')

# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3ec2nd.dev.out', 'dd3ec2nd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3ec2nd.dev.out', 'dd3ec2nd')
# print('')

# print('--------CN TST--------')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\2nd.tst.out', '  2nd')
# print('')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\ec2nd.tst.out', 'ec2nd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\ec2nd.tst.out', 'ec2nd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out', 'dd2nd')
# print('------------')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\2nd.tst.out', '   2nd')
# print('')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\ec2ndf.tst.out', 'ec2ndf')
# eval_empty_node('cn.tst.tree', 'cn_feat\\ec2ndf.tst.out', 'ec2ndf')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\comb2ndf.tst.out', 'comb2ndf')
# eval_empty_node('cn.tst.tree', 'cn_feat\\comb2ndf.tst.out', 'comb2ndf')
# eval_empty_tree('cn_feat\\dd2ndf.tst.out', 'cn_feat\\comb2ndf.tst.out', 'similarity')
# eval_empty_node('cn_feat\\dd2ndf.tst.out', 'cn_feat\\comb2ndf.tst.out', 'similarity')
# print('------------')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\3rd.tst.out', '  3rd')
# print('')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out', 'dd3rd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out', 'dd3rd')
# print('------------')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ec3rd.tst.out', 'dd2ec3rd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ec3rd.tst.out', 'dd2ec3rd')
# print('------------')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3ec2nd.tst.out', 'dd3ec2nd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3ec2nd.tst.out', 'dd3ec2nd')
# print('')

# print('--------EN DEV--------')
# eval_empty_tree('en.tst.tree', 'en_feat\\2nd.tst.out', '  2nd')
# print('')
# eval_empty_tree('en.tst.tree', 'en_feat\\ec2nd.tst.out', 'ec2nd')
# eval_empty_node('en.tst.tree', 'en_feat\\ec2nd.tst.out', 'ec2nd')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out', 'dd2nd')
# print('------------')

# eval_empty_tree('en.tst.tree', 'en_feat\\2nd.tst.out', '   2nd')
# print('')
# eval_empty_tree('en.tst.tree', 'en_feat\\ec2ndf.tst.out', 'ec2ndf')
# eval_empty_node('en.tst.tree', 'en_feat\\ec2ndf.tst.out', 'ec2ndf')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_tree('en.tst.tree', 'en_feat\\comb2ndf.tst.out', 'comb2ndf')
# eval_empty_node('en.tst.tree', 'en_feat\\comb2ndf.tst.out', 'comb2ndf')
# eval_empty_tree('en_feat\\dd2ndf.tst.out', 'en_feat\\comb2ndf.tst.out', 'similarity')
# eval_empty_node('en_feat\\dd2ndf.tst.out', 'en_feat\\comb2ndf.tst.out', 'similarity')
# print('------------')

# eval_empty_tree('en.tst.tree', 'en_feat\\3rd.tst.out', '  3rd')
# print('')
# eval_empty_tree('en.tst.tree', 'en_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_node('en.tst.tree', 'en_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out', 'dd3rd')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out', 'dd3rd')
# print('------------')

# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ec3rd.tst.out', 'dd2ec3rd')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ec3rd.tst.out', 'dd2ec3rd')
# print('------------')

# eval_empty_tree('en.tst.tree', 'en_feat\\dd3ec2nd.tst.out', 'dd3ec2nd')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3ec2nd.tst.out', 'dd3ec2nd')
# print('')

# print('--------EN TST--------')
# eval_empty_tree('en.dev.tree', 'en_feat\\2nd.dev.out', '  2nd')
# print('')
# eval_empty_tree('en.dev.tree', 'en_feat\\ec2nd.dev.out', 'ec2nd')
# eval_empty_node('en.dev.tree', 'en_feat\\ec2nd.dev.out', 'ec2nd')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out', 'dd2nd')
# print('------------')

# eval_empty_tree('en.dev.tree', 'en_feat\\2nd.dev.out', '   2nd')
# print('')
# eval_empty_tree('en.dev.tree', 'en_feat\\ec2ndf.dev.out', 'ec2ndf')
# eval_empty_node('en.dev.tree', 'en_feat\\ec2ndf.dev.out', 'ec2ndf')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_tree('en.dev.tree', 'en_feat\\comb2ndf.dev.out', 'comb2ndf')
# eval_empty_node('en.dev.tree', 'en_feat\\comb2ndf.dev.out', 'comb2ndf')
# eval_empty_tree('en_feat\\dd2ndf.dev.out', 'en_feat\\comb2ndf.dev.out', 'similarity')
# eval_empty_node('en_feat\\dd2ndf.dev.out', 'en_feat\\comb2ndf.dev.out', 'similarity')
# print('------------')

# eval_empty_tree('en.dev.tree', 'en_feat\\3rd.dev.out', '  3rd')
# print('')
# eval_empty_tree('en.dev.tree', 'en_feat\\ec3rd.dev.out', 'ec3rd')
# eval_empty_node('en.dev.tree', 'en_feat\\ec3rd.dev.out', 'ec3rd')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out', 'dd3rd')
# print('------------')

# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ec3rd.dev.out', 'dd2ec3rd')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ec3rd.dev.out', 'dd2ec3rd')
# print('------------')

# eval_empty_tree('en.dev.tree', 'en_feat\\dd3ec2nd.dev.out', 'dd3ec2nd')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3ec2nd.dev.out', 'dd3ec2nd')
# print('')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\2nd.tst.out', '2nd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\ec2nd.tst.out', 'ec2nd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.1', 'dd2nd0.1')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.1', 'dd2nd0.1')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.2', 'dd2nd0.2')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.2', 'dd2nd0.2')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.3', 'dd2nd0.3')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.3', 'dd2nd0.3')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.4', 'dd2nd0.4')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.4', 'dd2nd0.4')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.6', 'dd2nd0.6')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.6', 'dd2nd0.6')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.7', 'dd2nd0.7')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.7', 'dd2nd0.7')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.8', 'dd2nd0.8')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.8', 'dd2nd0.8')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.9', 'dd2nd0.9')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out0.9', 'dd2nd0.9')


# eval_empty_tree('cn.dev.tree', 'cn_feat\\2nd.dev.out', '2nd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\ec2nd.dev.out', 'ec2nd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.1', 'dd2nd0.1')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.1', 'dd2nd0.1')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.2', 'dd2nd0.2')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.2', 'dd2nd0.2')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.3', 'dd2nd0.3')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.3', 'dd2nd0.3')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.4', 'dd2nd0.4')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.4', 'dd2nd0.4')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.6', 'dd2nd0.6')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.6', 'dd2nd0.6')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.7', 'dd2nd0.7')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.7', 'dd2nd0.7')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.8', 'dd2nd0.8')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.8', 'dd2nd0.8')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.9', 'dd2nd0.9')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out0.9', 'dd2nd0.9')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\2nd.tst.out', '2nd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\ec2ndf.tst.out', 'ec2ndf')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.1', 'dd2ndf0.1')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.1', 'dd2ndf0.1')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.2', 'dd2ndf0.2')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.2', 'dd2ndf0.2')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.3', 'dd2ndf0.3')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.3', 'dd2ndf0.3')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.4', 'dd2ndf0.4')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.4', 'dd2ndf0.4')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.6', 'dd2ndf0.6')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.6', 'dd2ndf0.6')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.7', 'dd2ndf0.7')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.7', 'dd2ndf0.7')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.8', 'dd2ndf0.8')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.8', 'dd2ndf0.8')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.9', 'dd2ndf0.9')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out0.9', 'dd2ndf0.9')


# eval_empty_tree('cn.dev.tree', 'cn_feat\\2nd.dev.out', '2nd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\ec2ndf.dev.out', 'ec2ndf')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.1', 'dd2ndf0.1')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.1', 'dd2ndf0.1')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.2', 'dd2ndf0.2')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.2', 'dd2ndf0.2')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.3', 'dd2ndf0.3')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.3', 'dd2ndf0.3')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.4', 'dd2ndf0.4')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.4', 'dd2ndf0.4')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.6', 'dd2ndf0.6')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.6', 'dd2ndf0.6')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.7', 'dd2ndf0.7')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.7', 'dd2ndf0.7')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.8', 'dd2ndf0.8')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.8', 'dd2ndf0.8')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.9', 'dd2ndf0.9')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out0.9', 'dd2ndf0.9')


# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\comb2nd.dev.out', 'comb2nd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\comb2nd.dev.out', 'comb2nd')
# eval_empty_tree('cn_feat\\comb2nd.dev.out', 'cn_feat\\dd2nd.dev.out', 'compare')
# eval_empty_node('cn_feat\\comb2nd.dev.out', 'cn_feat\\dd2nd.dev.out', 'compare')

# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\comb2ndf.dev.out', 'comb2ndf')
# eval_empty_node('cn.dev.tree', 'cn_feat\\comb2ndf.dev.out', 'comb2ndf')
# eval_empty_tree('cn_feat\\comb2ndf.dev.out', 'cn_feat\\dd2ndf.dev.out', 'compare')
# eval_empty_node('cn_feat\\comb2ndf.dev.out', 'cn_feat\\dd2ndf.dev.out', 'compare')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\comb2nd.tst.out', 'comb2nd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\comb2nd.tst.out', 'comb2nd')
# eval_empty_tree('cn_feat\\comb2nd.tst.out', 'cn_feat\\dd2nd.tst.out', 'compare')
# eval_empty_node('cn_feat\\comb2nd.tst.out', 'cn_feat\\dd2nd.tst.out', 'compare')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\comb2ndf.tst.out', 'comb2ndf')
# eval_empty_node('cn.tst.tree', 'cn_feat\\comb2ndf.tst.out', 'comb2ndf')
# eval_empty_tree('cn_feat\\comb2ndf.tst.out', 'cn_feat\\dd2ndf.tst.out', 'compare')
# eval_empty_node('cn_feat\\comb2ndf.tst.out', 'cn_feat\\dd2ndf.tst.out', 'compare')

# eval_empty_tree('en.tst.tree', 'en_feat\\2nd.tst.out', '2nd')
# eval_empty_tree('en.tst.tree', 'en_feat\\ec2nd.tst.out', 'ec2nd')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out0.1', 'dd2nd0.1')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out0.1', 'dd2nd0.1')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out0.2', 'dd2nd0.2')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out0.2', 'dd2nd0.2')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out0.3', 'dd2nd0.3')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out0.3', 'dd2nd0.3')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out0.4', 'dd2nd0.4')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out0.4', 'dd2nd0.4')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out0.6', 'dd2nd0.6')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out0.6', 'dd2nd0.6')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out0.7', 'dd2nd0.7')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out0.7', 'dd2nd0.7')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out0.8', 'dd2nd0.8')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out0.8', 'dd2nd0.8')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out0.9', 'dd2nd0.9')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out0.9', 'dd2nd0.9')


# eval_empty_tree('en.dev.tree', 'en_feat\\2nd.dev.out', '2nd')
# eval_empty_tree('en.dev.tree', 'en_feat\\ec2nd.dev.out', 'ec2nd')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out0.1', 'dd2nd0.1')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out0.1', 'dd2nd0.1')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out0.2', 'dd2nd0.2')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out0.2', 'dd2nd0.2')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out0.3', 'dd2nd0.3')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out0.3', 'dd2nd0.3')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out0.4', 'dd2nd0.4')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out0.4', 'dd2nd0.4')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out0.6', 'dd2nd0.6')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out0.6', 'dd2nd0.6')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out0.7', 'dd2nd0.7')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out0.7', 'dd2nd0.7')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out0.8', 'dd2nd0.8')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out0.8', 'dd2nd0.8')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out0.9', 'dd2nd0.9')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out0.9', 'dd2nd0.9')

# eval_empty_tree('en.tst.tree', 'en_feat\\2nd.tst.out', '2nd')
# eval_empty_tree('en.tst.tree', 'en_feat\\ec2ndf.tst.out', 'ec2ndf')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.1', 'dd2ndf0.1')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.1', 'dd2ndf0.1')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.2', 'dd2ndf0.2')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.2', 'dd2ndf0.2')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.3', 'dd2ndf0.3')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.3', 'dd2ndf0.3')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.4', 'dd2ndf0.4')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.4', 'dd2ndf0.4')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.6', 'dd2ndf0.6')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.6', 'dd2ndf0.6')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.7', 'dd2ndf0.7')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.7', 'dd2ndf0.7')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.8', 'dd2ndf0.8')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.8', 'dd2ndf0.8')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.9', 'dd2ndf0.9')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out0.9', 'dd2ndf0.9')


# eval_empty_tree('en.dev.tree', 'en_feat\\2nd.dev.out', '2nd')
# eval_empty_tree('en.dev.tree', 'en_feat\\ec2ndf.dev.out', 'ec2ndf')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.1', 'dd2ndf0.1')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.1', 'dd2ndf0.1')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.2', 'dd2ndf0.2')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.2', 'dd2ndf0.2')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.3', 'dd2ndf0.3')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.3', 'dd2ndf0.3')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.4', 'dd2ndf0.4')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.4', 'dd2ndf0.4')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.6', 'dd2ndf0.6')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.6', 'dd2ndf0.6')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.7', 'dd2ndf0.7')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.7', 'dd2ndf0.7')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.8', 'dd2ndf0.8')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.8', 'dd2ndf0.8')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.9', 'dd2ndf0.9')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out0.9', 'dd2ndf0.9')

# eval_empty_tree('en.dev.tree', 'en_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2nd.dev.out', 'dd2nd')
# eval_empty_tree('en.dev.tree', 'en_feat\\comb2nd.dev.out', 'comb2nd')
# eval_empty_node('en.dev.tree', 'en_feat\\comb2nd.dev.out', 'comb2nd')
# eval_empty_tree('en_feat\\comb2nd.dev.out', 'en_feat\\dd2nd.dev.out', 'compare')
# eval_empty_node('en_feat\\comb2nd.dev.out', 'en_feat\\dd2nd.dev.out', 'compare')

# eval_empty_tree('en.dev.tree', 'en_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_node('en.dev.tree', 'en_feat\\dd2ndf.dev.out', 'dd2ndf')
# eval_empty_tree('en.dev.tree', 'en_feat\\comb2ndf.dev.out', 'comb2ndf')
# eval_empty_node('en.dev.tree', 'en_feat\\comb2ndf.dev.out', 'comb2ndf')
# eval_empty_tree('en_feat\\comb2ndf.dev.out', 'en_feat\\dd2ndf.dev.out', 'compare')
# eval_empty_node('en_feat\\comb2ndf.dev.out', 'en_feat\\dd2ndf.dev.out', 'compare')

# eval_empty_tree('en.tst.tree', 'en_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2nd.tst.out', 'dd2nd')
# eval_empty_tree('en.tst.tree', 'en_feat\\comb2nd.tst.out', 'comb2nd')
# eval_empty_node('en.tst.tree', 'en_feat\\comb2nd.tst.out', 'comb2nd')
# eval_empty_tree('en_feat\\comb2nd.tst.out', 'en_feat\\dd2nd.tst.out', 'compare')
# eval_empty_node('en_feat\\comb2nd.tst.out', 'en_feat\\dd2nd.tst.out', 'compare')

# eval_empty_tree('en.tst.tree', 'en_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_node('en.tst.tree', 'en_feat\\dd2ndf.tst.out', 'dd2ndf')
# eval_empty_tree('en.tst.tree', 'en_feat\\comb2ndf.tst.out', 'comb2ndf')
# eval_empty_node('en.tst.tree', 'en_feat\\comb2ndf.tst.out', 'comb2ndf')
# eval_empty_tree('en_feat\\comb2ndf.tst.out', 'en_feat\\dd2ndf.tst.out', 'compare')
# eval_empty_node('en_feat\\comb2ndf.tst.out', 'en_feat\\dd2ndf.tst.out', 'compare')



# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\comb3rd.dev.out', 'comb3rd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\comb3rd.dev.out', 'comb3rd')
# eval_empty_tree('cn_feat\\comb3rd.dev.out', 'cn_feat\\dd3rd.dev.out', 'compare')
# eval_empty_node('cn_feat\\comb3rd.dev.out', 'cn_feat\\dd3rd.dev.out', 'compare')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out', 'dd3rd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out', 'dd3rd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\comb3rd.tst.out', 'comb3rd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\comb3rd.tst.out', 'comb3rd')
# eval_empty_tree('cn_feat\\comb3rd.tst.out', 'cn_feat\\dd3rd.tst.out', 'compare')
# eval_empty_node('cn_feat\\comb3rd.tst.out', 'cn_feat\\dd3rd.tst.out', 'compare')

# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_tree('en.dev.tree', 'en_feat\\comb3rd.dev.out', 'comb3rd')
# eval_empty_node('en.dev.tree', 'en_feat\\comb3rd.dev.out', 'comb3rd')
# eval_empty_tree('en_feat\\comb3rd.dev.out', 'en_feat\\dd3rd.dev.out', 'compare')
# eval_empty_node('en_feat\\comb3rd.dev.out', 'en_feat\\dd3rd.dev.out', 'compare')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\3rd.tst.out', '3rd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.1', 'dd3rd0.1')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.1', 'dd3rd0.1')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.2', 'dd3rd0.2')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.2', 'dd3rd0.2')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.3', 'dd3rd0.3')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.3', 'dd3rd0.3')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.4', 'dd3rd0.4')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.4', 'dd3rd0.4')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out', 'dd3rd')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out', 'dd3rd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.6', 'dd3rd0.6')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.6', 'dd3rd0.6')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.7', 'dd3rd0.7')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.7', 'dd3rd0.7')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.8', 'dd3rd0.8')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.8', 'dd3rd0.8')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.9', 'dd3rd0.9')
# eval_empty_node('cn.tst.tree', 'cn_feat\\dd3rd.tst.out0.9', 'dd3rd0.9')

# eval_empty_tree('cn.dev.tree', 'cn_feat\\3rd.dev.out', '3rd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\ec3rd.dev.out', 'ec3rd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.1', 'dd3rd0.1')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.1', 'dd3rd0.1')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.2', 'dd3rd0.2')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.2', 'dd3rd0.2')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.3', 'dd3rd0.3')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.3', 'dd3rd0.3')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.4', 'dd3rd0.4')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.4', 'dd3rd0.4')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.6', 'dd3rd0.6')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.6', 'dd3rd0.6')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.7', 'dd3rd0.7')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.7', 'dd3rd0.7')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.8', 'dd3rd0.8')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.8', 'dd3rd0.8')
# eval_empty_tree('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.9', 'dd3rd0.9')
# eval_empty_node('cn.dev.tree', 'cn_feat\\dd3rd.dev.out0.9', 'dd3rd0.9')

# eval_empty_tree('en.dev.tree', 'en_feat\\3rd.dev.out', '3rd')
# eval_empty_tree('en.dev.tree', 'en_feat\\ec3rd.dev.out', 'ec3rd')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out0.1', 'dd3rd0.1')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out0.1', 'dd3rd0.1')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out0.2', 'dd3rd0.2')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out0.2', 'dd3rd0.2')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out0.3', 'dd3rd0.3')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out0.3', 'dd3rd0.3')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out0.4', 'dd3rd0.4')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out0.4', 'dd3rd0.4')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out', 'dd3rd')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out0.6', 'dd3rd0.6')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out0.6', 'dd3rd0.6')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out0.7', 'dd3rd0.7')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out0.7', 'dd3rd0.7')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out0.8', 'dd3rd0.8')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out0.8', 'dd3rd0.8')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out0.9', 'dd3rd0.9')
# eval_empty_node('en.dev.tree', 'en_feat\\dd3rd.dev.out0.9', 'dd3rd0.9')

# eval_empty_tree('en.tst.tree', 'en_feat\\3rd.tst.out', '3rd')
# eval_empty_tree('en.tst.tree', 'en_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out0.1', 'dd3rd0.1')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out0.1', 'dd3rd0.1')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out0.2', 'dd3rd0.2')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out0.2', 'dd3rd0.2')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out0.3', 'dd3rd0.3')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out0.3', 'dd3rd0.3')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out0.4', 'dd3rd0.4')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out0.4', 'dd3rd0.4')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out', 'dd3rd')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out', 'dd3rd')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out0.6', 'dd3rd0.6')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out0.6', 'dd3rd0.6')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out0.7', 'dd3rd0.7')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out0.7', 'dd3rd0.7')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out0.8', 'dd3rd0.8')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out0.8', 'dd3rd0.8')
# eval_empty_tree('en.tst.tree', 'en_feat\\dd3rd.tst.out0.9', 'dd3rd0.9')
# eval_empty_node('en.tst.tree', 'en_feat\\dd3rd.tst.out0.9', 'dd3rd0.9')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\2nd.tst.out', '2nd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\3rd.tst.out', '3rd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_tree('en.tst.tree', 'en_feat\\2nd.tst.out', '2nd')
# eval_empty_tree('en.tst.tree', 'en_feat\\3rd.tst.out', '3rd')
# eval_empty_tree('en.tst.tree', 'en_feat\\ec3rd.tst.out', 'ec3rd')

# eval_empty_tree('en.dev.tree', 'en_feat/3rd.dev.out', '3rd')
# eval_empty_tree('en_feat\\3rd.dev.out', 'en_feat\\ec3rd.dev.out', 'ec3rd')
# eval_empty_tree('en.dev.tree', 'tmp', '3rd')
# eval_empty_tree('en.dev.tree', 'en_feat\\dd3rd.dev.out', 'dd3rd')

# eval_empty_tree('cn.tst.tree', 'cn_feat\\3rd.tst.out', '3rd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_tree('cn_feat\\3rd.tst.out', 'cn_feat\\ec3rd.tst.out', 'ec3rd')
# eval_empty_tree('cn.tst.tree', 'cn_feat\\dd3rd.tst.out', 'dd3rd')

if __name__ == "__main__":
	print(eval_empty_tree(sys.argv[1], sys.argv[2], 'tree'))
	# print('----EN TST----')
	# eval_empty_tree('en.dev.tree', 'en_feat/2nd.dev.out', '2nd')
	# eval_empty_tree('en.dev.tree', 'en_feat/ec2nd.dev.out', 'ec2nd')
	# eval_empty_tree('en.dev.tree', 'en_feat/dd2nd.dev.out', 'dd2nd')
	# eval_empty_tree('en.dev.tree', 'en_feat/comb2nd.dev.out', 'comb2nd')
	# eval_empty_node('en.dev.tree', 'en_feat/comb2nd.dev.out', 'comb2nd')
	# eval_empty_tree('en_feat/dd2nd.dev.out', 'en_feat/comb2nd.dev.out', 'ddcomb')
	# eval_empty_tree('en.dev.tree', 'en_feat/ec2ndf.dev.out', 'ec2ndf')
	# eval_empty_tree('en.dev.tree', 'en_feat/dd2ndf.dev.out', 'dd2ndf')
	# eval_empty_tree('en.dev.tree', 'en_feat/comb2ndf.dev.out', 'comb2ndf')
	# eval_empty_tree('en_feat/dd2ndf.dev.out', 'en_feat/comb2ndf.dev.out', 'ddcomb')
	# print('----EN DEV----')
	# eval_empty_tree('en.tst.tree', 'en_feat/2nd.tst.out', '2nd')
	# eval_empty_tree('en.tst.tree', 'en_feat/ec2nd.tst.out', 'ec2nd')
	# eval_empty_tree('en.tst.tree', 'en_feat/dd2nd.tst.out', 'dd2nd')
	# eval_empty_tree('en.tst.tree', 'en_feat/comb2nd.tst.out', 'comb2nd')
	# eval_empty_node('en.tst.tree', 'en_feat/comb2nd.tst.out', 'comb2nd')
	# eval_empty_tree('en_feat/dd2nd.tst.out', 'en_feat/comb2nd.tst.out', 'ddcomb')
	# eval_empty_tree('en.tst.tree', 'en_feat/ec2ndf.tst.out', 'ec2ndf')
	# eval_empty_tree('en.tst.tree', 'en_feat/dd2ndf.tst.out', 'dd2ndf')
	# eval_empty_node('en.tst.tree', 'en_feat/dd2ndf.tst.out', 'dd2ndf')
	# eval_empty_tree('en.tst.tree', 'en_feat/comb2ndf.tst.out', 'comb2ndf')
	# eval_empty_node('en.tst.tree', 'en_feat/comb2ndf.tst.out', 'comb2ndf')
	# eval_empty_tree('en_feat/dd2ndf.tst.out', 'en_feat/comb2ndf.tst.out', 'ddcomb')
