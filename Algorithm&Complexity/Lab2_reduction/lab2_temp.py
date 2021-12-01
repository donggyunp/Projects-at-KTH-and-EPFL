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
node_variable=[]

for ind,cha in enumerate(edges_1x[0]):
    if cha == " ":
        node_x1 = int(edges_1x[0][ind+1:])
for ind,cha in enumerate(edges_1x[1]):
    if cha == " ":
        node_x2 = int(edges_1x[1][ind+1:])
#node_variable.append(node_x1)
#node_variable.append(node_x2)
#print(node_x1,node_x2)

for i in range(200):    
    node_variable.append(node_x1)
    node_variable.append(node_x2)
    #print(node_variable)
    if node_x1 == 1:
        print('break!!!!!!!!!!!!!!!!!!!!!!!')
        break
    else:
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
        print(node_y) 
        node_x1 = int(node_y[0])
        if node_x1 != 1:            
            node_x2 = int(node_y[1])
print(node_variable[:-1])#[2, 10, 11, 16, 17, 22, 1]

###count the number of nodes in each variable
#if the neighboring node is connected to in both way, count +1
k = len(node_variable) -1
'''
for index,node in enumerate(node_variable):
    if node_variable[index +1] != 
    for i in range(5000):
        if int(node) == 
'''    
node_1=[]
node_2=[]
for n in edges:
    for ind,cha in enumerate(n):
        if cha == " ":
            node_1.append(int(n[0:ind]))
            node_2.append(int(n[ind+1:]))

#print(node_1[0],type(node_1[0]))
count =0
for ind1,node1 in enumerate(node_1):
    for ind2,node2 in enumerate(node_2):
        if node1 == node2 and node_1[ind2] == node_2[ind1]:
            count= count +1
print(count) 
'''
for index,node in enumerate(node_variable):
    for i in node_1:
        if node == i and :
            
'''            
