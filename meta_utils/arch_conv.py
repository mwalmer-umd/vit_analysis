def letter2arch(l):
	d = {
		'T': 'tiny',
		'S': 'small',
		'B': 'base',
		'L': 'large',
		'H': 'huge'
	}
	if l not in d:
		print('ERROR: invalid arch letter: ' + l)
		exit(-1)
	return d[l]