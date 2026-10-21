import re,glob,collections,sys
order=["content/ch1.txt","content/ch2a.txt","content/ch2b.txt","content/ch3a.txt","content/ch3b.txt","ch4_verify.py","ch4_phantom.py","ch4_hard.py","ch4_brats.py","content/ch5.txt","content/ch6.txt","appendices.py"]
cnt=collections.Counter(); first={}
loc=collections.defaultdict(list)
for f in order:
    try: s=open(f).read()
    except FileNotFoundError: continue
    for i,line in enumerate(s.split("\n"),1):
        for m in re.finditer(r"\[(@[A-Za-z0-9_]+(?:\s*;\s*@[A-Za-z0-9_]+)*)\]",line):
            for k in re.findall(r"@([A-Za-z0-9_]+)",m.group(1)):
                cnt[k]+=1; loc[k].append(f"{f}:{i}")
for k,v in cnt.most_common():
    if v>=int(sys.argv[1] if len(sys.argv)>1 else 5): print(v,k,loc[k][:12] if len(sys.argv)>2 else "")
print(len(cnt),"distinct")
