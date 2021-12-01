import sys                                                                  
import fileinput

def readinput():
    para = []
    for line in sys.stdin:
        para.append(line.rstrip())
    return para

topol = readinput()
edges = topol[3:]
V_num = int(topol[0])
E_num = int(topol[1])
m = int(topol[2])

role = V_num+2
scene = E_num + 2*V_num
actor = m+2
if actor > role:
    actor = role
U = list(range(1, actor+1))
for i in range(V_num):
    string1=str(V_num+1) + " " + str(i+1)
    string2=str(V_num+2) + " " + str(i+1)
    #print(string)
    edges.append(string1)
    edges.append(string2)

print(role)
print(scene)
print(actor)

for i in range(int(role)):
    print(len(U), *U)
for edge in edges:
    print(str(2) + " " + edge)
