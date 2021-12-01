import sys
import fileinput


def readinput():
    para=[]
    for line in sys.stdin:
        para.append(line.rstrip())
    return para

topo = readinput()
n = int(topo[0])
s = int(topo[1])
k = int(topo[2])

edgesss = topo[3:]
edges=[]
#n = int(input ())
#s = int(input ())
#k = int(input ())

#Edge = []
for edge in edgesss:
    start = ''
    for ind,char in enumerate(edge):
        if char != ' ':
            start += edge[ind]
        else:
            break
    edges.append([int(start),int(edge[ind:])])
'''
for i in range(s):
    x , y = input().split()
    edges.append([int(x),int(y)])
'''
#print()
#print(n)
#print(s)
#print(k)
#print(edges)

for color in range(k):

    visited_nodes = []
    utilised_colors = []
    const_1 = [[]]*n
    #const_1 = [[] for i in range(n)]

    for edge in edges:

        if edge[0] not in visited_nodes and edge[1] not in visited_nodes:
            visited_nodes.append(edge[0])
            visited_nodes.append(edge[1])
            const_1[edge[0]-1]= [color]
            #print()
            #print('TEST')
            #print(const_1)
            #print(visited_nodes)
            utilised_colors.append(color)

            if color == 1 or color == 2:
                const_1[edge[1]-1]= list(range(3, k+1))
                if len(const_1[edge[1]-1])==1:
                    if const_1[edge[1]-1] not in utilised_colors:
                        utilised_colors.append(const_1[edge[1]-1][0])
            else:
                const_1[edge[1]-1]= list(range(1, color))+ list(range(color+1,k+1))
                if len(const_1[edge[1]-1])==1:
                    if const_1[edge[1]-1] not in utilised_colors:
                        utilised_colors.append(const_1[edge[1]-1][0])

        elif edge[0] in visited_nodes and edge[1] not in visited_nodes:
            visited_nodes.append(edge[1])
            if len(const_1[edge[0]-1])== 1:
                color_neigh = const_1[edge[0]-1][0]
                if color_neigh == 1 or color_neigh == 2:
                    const_1[edge[1]-1]= list(range(3, k+1))
                    if len(const_1[edge[1]-1])==1:
                        if const_1[edge[1]-1] not in utilised_colors:
                            utilised_colors.append(const_1[edge[1]-1][0])
                else:
                    if 1 in utilised_colors:
                        #const_1[edge[1]-1]= list(range(1, color_neigh)) + list(range(color_neigh+1,k))

                        const_1[edge[1]-1]= list(range(2, k+1))
                    #const_1[edge[1]-1].remove(1)
                        const_1[edge[1]-1].remove(color_neigh)


                    if 2 in utilised_colors:
                        #temp=[0]
                        #const_1[edge[1]-1]= temp.extend([range(2, color_neigh)])
                        #const_1[edge[1]-1].extend([range(color_neigh+1,k)])

                        const_1[edge[1]-1]= list(range(1, k+1))
                        const_1[edge[1]-1].remove(2)
                        const_1[edge[1]-1].remove(color_neigh)

                        #extendlist(range(color_neigh+1,k))
                        #const_1[edge[1]-1]= list(0 + range(2, color_neigh)) + list(range(color_neigh+1,k))
                    #const_1[edge[1]]= list(range(0, color_neigh))+ list(range(color_neigh+1,k))

                    if len(const_1[edge[1]-1])==1:
                        if const_1[edge[1]-1][0] not in utilised_colors:
                            utilised_colors.append(const_1[edge[1]-1][0])
            else:
                const_1[edge[1]-1]= list(range(1, k+1))

        elif edge[0] not in visited_nodes and edge[1] in visited_nodes:
            visited_nodes.append(edge[0])
            if len(const_1[edge[1]-1])== 1:
                color_neigh = const_1[edge[1]-1][0]
                if color_neigh == 0 or color_neigh == 1:
                    const_1[edge[0]-1]= list(range(2, k+1))
                    if len(const_1[edge[0]-1])==1:
                        if const_1[edge[0]-1][0] not in utilised_colors:
                            utilised_colors.append(const_1[edge[0]-1][0])
                else:
                    if 1 in utilised_colors:
                        #const_1[edge[0]-1]= list(range(1, color_neigh)) + list(range(color_neigh+1,k))

                        const_1[edge[0]-1]= list(range(2, k+1))
                    #const_1[edge[1]-1].remove(1)
                        const_1[edge[0]-1].remove(color_neigh)


                    if 2 in utilised_colors:
                        #const_1[edge[0]-1]= list(0 + range(2, color_neigh)) + list(range(color_neigh+1,k))

                        const_1[edge[0]-1]= list(range(1, k+1))
                        const_1[edge[0]-1].remove(2)
                        const_1[edge[0]-1].remove(color_neigh)

                    if len(const_1[edge[0]-1])==1:
                        if const_1[edge[0]-1] not in utilised_colors:
                            utilised_colors.append(const_1[edge[0]-1][0])
            else:
                const_1[edge[0]-1]= list(range(1, k+1))

#    print()
#    print('fine ciclo color')
#    print(utilised_colors)
#    print(const_1)

    if len(utilised_colors)==k:
        break
#    if 0 in utilised_colors and 1 in utilised_colors:
#        break
#    else:



if len(visited_nodes) < n:
    eliminate = n - len(visited_nodes)
    n = n - eliminate

#if len(utilised_colors) < ker:

#print()
print(n)
print(s)
print(k)
for i in range(len(const_1)):
    if len(const_1[i])==0:
        continue
    print(len(const_1[i]), int(*const_1[i])+1)
for edge in edges:
    print('2', *edge)
#print(const_1)




#def readinput():
#    para = []
#    for line in sys.stdin:
#        para.append(line.rstrip())
#    return para

#topol = readinput()
#print(topol)

#edges = topol[2:]
#print(edges)
#print(edges[0][0])
#edges_1x=[]
#edges_xy=[]
#edges_yz=[]
#edges_z1=[]

#get edges from node 1 to x
#for x in edges:
#    start = ""
#    end = ""
#    for ind,char in enumerate(x):
#        if char != " ":
#            start += x[ind]
#        else:
#            break
#    start =int(start)
#    end = int(x[ind:])
#    if start == 1:
#        edges_1x.append(x)
#print(edges_1x) #['1 2', '1 10']
#for ind,cha in enumerate(edges_1x[0]):
#    if cha == " ":
#        node_x1 = int(edges_1x[0][ind+1:])
#for ind,cha in enumerate(edges_1x[1]):
#    if cha == " ":
#        node_x2 = int(edges_1x[1][ind+1:])
#print(node_x1,node_x2)

#get edges from node x to y
#temp1=[]
#temp2=[]
#for edge in edges:
#    for ind,cha in enumerate(edge):
#        if cha == " ":
#            node_1 = edge[0:ind]
#            node_2 = edge[ind+1:]
#    if str(node_x1) == node_1:
#        temp1.append(edge)
#    if str(node_x2) == node_1:
#        temp2.append(edge)
#print(temp1,temp2)#['2 3', '2 11', '2 16'] ['10 9', '10 11', '10 16']
#node_y=[]
#for edge1 in temp1:
#    for ind,cha in enumerate(edge1):
#        if cha == " ":
#            node_1in1 = edge1[0:ind]
#            node_2in1 = edge1[ind+1:]
#    for edge2 in temp2:
#        for ind,cha in enumerate(edge2):
#            if cha == " ":
#                node_1in2 = edge2[0:ind]
#                node_2in2 = edge2[ind+1:]
#        if node_2in1 == node_2in2:
#            node_y.append(node_2in2)

#node_y1 = int(node_y[0])
#node_y2 = int(node_y[1])
#print(node_y1,node_y2)
#node_z=[]
#for edge in edges:
#    for ind,cha in enumerate(edge):
#        if cha == " ":
#            node_1 = edge[0:ind]
#            node_2 = edge[ind+1:]
#    if node_2 == '1':
#        node_z.append(node_1)
#node_z1 = int(node_z[0])
#node_z2 = int(node_z[1])
#print(node_z1,node_z2)
