import sys                                                                  
import fileinput

def readinput():
    para = []
    for line in sys.stdin:
        para.append(line.rstrip())
    return para

topol = readinput()
#print(topol
edges = topol[2:]
#print(edges)
#print(edges[0][0])
edges_1x=[]
edges_xy=[]
edges_yz=[]
edges_z1=[]

#get edges from node 1 to x
for x in edges:
    start = ""
    end = ""
    for ind,char in enumerate(x):
        if char != " ":
            start += x[ind]
        else:
            break
    start =int(start)
    end = int(x[ind:])
    if start == 1:
        edges_1x.append(x)
#print(edges_1x) #['1 2', '1 10']
for ind,cha in enumerate(edges_1x[0]):
    if cha == " ":
        node_x1 = int(edges_1x[0][ind+1:])
for ind,cha in enumerate(edges_1x[1]):
    if cha == " ":
        node_x2 = int(edges_1x[1][ind+1:])
print(node_x1,node_x2)

#get edges from node x to y
temp1=[]
temp2=[]
for edge in edges:
    for ind,cha in enumerate(edge):
        if cha == " ":
            node_1 = edge[0:ind]
            node_2 = edge[ind+1:]
    if str(node_x1) == node_1:
        temp1.append(edge) 
    if str(node_x2) == node_1:
        temp2.append(edge)
#print(temp1,temp2)#['2 3', '2 11', '2 16'] ['10 9', '10 11', '10 16']
node_y=[]
for edge1 in temp1:
    for ind,cha in enumerate(edge1):
        if cha == " ":
            node_1in1 = edge1[0:ind]
            node_2in1 = edge1[ind+1:]
    for edge2 in temp2:
        for ind,cha in enumerate(edge2):
            if cha == " ":
                node_1in2 = edge2[0:ind]
                node_2in2 = edge2[ind+1:]
        if node_2in1 == node_2in2:
            node_y.append(node_2in2)
        
node_y1 = int(node_y[0])
node_y2 = int(node_y[1])
print(node_y1,node_y2)
node_z=[]
for edge in edges:
    for ind,cha in enumerate(edge):
        if cha == " ":
            node_1 = edge[0:ind]
            node_2 = edge[ind+1:]
    if node_2 == '1':
        node_z.append(node_1)
node_z1 = int(node_z[0])
node_z2 = int(node_z[1])
print(node_z1,node_z2)



