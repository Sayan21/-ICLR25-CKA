import sys
f = open(sys.argv[1])

lines = f.readlines()
KLD = [0]*25
i=0
j=0
for l in lines:
    loss = l.split(",")

    if(i%2==1):
        sumloss += int(1000*float(loss[0].lstrip("[").strip("'")) + 1000*float(loss[1].strip(" ").strip("'")))
        KLD[j] = sumloss/2
        j+=1
    else:
        sumloss = int(1000*float(loss[0].lstrip("[").strip("'")) + 1000*float(loss[1].strip(" ").strip("'")))
    
    i+=1

    if(i==50): break

print(KLD)
f.close()


