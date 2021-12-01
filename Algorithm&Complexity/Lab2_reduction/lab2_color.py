import sys                                                                  
import fileinput
import random

def readinput():
    para = []
    for line in sys.stdin:
        para.append(line.rstrip())
    return para

topol = readinput()
edges = topol[3:]
V_num = topol[0]
E_num = topol[1]
m = topol[2]
print(V_num)
print(E_num)
print(m)

U = []
for i in range(int(m)):
    U.append(i+1)
###we might need to do it from the empty list



#print(U)
node1=[]
node2=[]
visited=[]
node_color = [[] for i in range(int(V_num))]
for edge in edges:
    start = ""
    for ind,char in enumerate(edge):
        if char != " ":
            start += edge[ind]
        else:
            break
    node1.append(int(start))
    node2.append(int(edge[ind:]))
    if node1 not in visited
        visited.append(node1)
        if colors == []
            
    visited.appned(node2)
    


print(node1)
print(node2)
#1. remove the sole vertex
sole_list=[]
for v in range(int(V_num)):
    if v not in node1 and v not in node2:
        sole_list.append(v)
print(sole_list)
V_inst = int(V_num) - len(sole_list)

#2. check if the strating color is correct = check the constraint for P1 and P2
# if 1 appears at least once and 2 appears at least onece and they are not in the same list


#3. create visited list of vertex 
visit=[]
for node in node1:
    visit.append(node)
for node in node2:
    visit.append(node)
visit = list(set(visit))

node_node = [[] for i in range(int(V_num))]
#node_color[0].append(0)
#print(node_color)

for i in range(int(E_num)):
    for j in range(int(V_num)):
        if node1[i]==j+1:
            temp = node_node[j]
            temp.append(node2[i])
        elif node2[i]==j+1:
            temp = node_node[j]
            temp.append(node1[i])
        else:
            pass
#print(node_node)#node_node[i] = [2,3,4] => node i is connected to node 2,3,4

node_color = [[] for i in range(int(V_num))]
for i in range(len(node_color)):
    node_color[i]=U
    #print(U)
#node_color[0] = [1]
#print(node_color)
no_no_uni=list(set(i for j in node_node for i in j))
print(no_no_uni)
for color in U:                                              
    node_color[0]=[color]
    for ind,line in enumerate(node_node):
        for node in line:
        #print("line in for:",line)
        #print("node in for:",node)
            #print(node_color[0])   
            if node_color[ind] == node_color[node-1]:
                node_color[ind].remove(random.choice(node_color[ind]))
    node_color_list = list(set(i for j in node_color for i in j))
    print("color:",color)
    if node_color_list == U: 
        print(node_color_list)
        print("break")
        break
            #check=all(item in U for item in node_color_list)
#for i in rage(len(node_color)):
'''
        if 1 in node_color[ind]:
            node_color[node-1].remove(1)
        if 2 in node_color[ind]:
            node_color[node-1].remove(2)
'''
'''
        if 2 in node_color[i+1]:
            node_color[i+1].remove(2)
        elif len(node_color[i]) == 1 and node_color[i][0] ==2:
            if 1 in node_color[i+1]:
                node_color[i+1].remove(1)
        elif len(node_color[i]) == 1 and node_color[i][0] != 1:
            node_color[i+1].remove(node_color[i][0])
    #elif len(node_color[i]) == 1 and node_color[i][0] != 2:
'''
for i in range(int(V_num)):
    
    #print(str(len(node_color[i])) + " " + str(node_color[i]))
    print(str(len(node_color[i])) ,*node_color[i])#use this line for final


#print 'EVERY SCENE'
for edge in edges:
    print(str(2) + " " + edge)
