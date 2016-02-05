
for lb in [0.01,0.05,0.1]:
	for s in [5,9,15]:
		s=str(s);
		lb=str(lb);
		print "./spc2 "+lb+ " "+s+ " >log_"+lb+"_"+s + " 2>err_"+lb+"_"+s+" &"