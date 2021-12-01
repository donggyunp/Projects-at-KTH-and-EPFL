import dtree as d
import monkdata as m
import random
import drawtree_qt5 as dt

m1 = m.monk1
m2 = m.monk2
m3 = m.monk3
m1test = m.monk1test
m2test = m.monk2test
m3test = m.monk3test

#def main():
entropy1 = d.entropy(m1)
  #entropy2 = d.entropy(m2)
  #entropy3 = d.entropy(m3)
  #entropy1test = d.entropy(m1test)
  #entropy2test = d.entropy(m1test)
  #entropy3test = d.entropy(m1test)
  #print (entropy1, entropy2, entropy3)
'''
  aG11 = d.averageGain(m1,m.attributes[0])
  aG12 = d.averageGain(m1,m.attributes[1])
  aG13 = d.averageGain(m1,m.attributes[2])
  aG14 = d.averageGain(m1,m.attributes[3])
  aG15 = d.averageGain(m1,m.attributes[4])
  aG16 = d.averageGain(m1,m.attributes[5])
  aG21 = d.averageGain(m2,m.attributes[0])
  aG22 = d.averageGain(m2,m.attributes[1])
  aG23 = d.averageGain(m2,m.attributes[2])
  aG24 = d.averageGain(m2,m.attributes[3])
  aG25 = d.averageGain(m2,m.attributes[4])
  aG26 = d.averageGain(m2,m.attributes[5])
  aG31 = d.averageGain(m3,m.attributes[0])
  aG32 = d.averageGain(m3,m.attributes[1])
  aG33 = d.averageGain(m3,m.attributes[2])
  aG34 = d.averageGain(m3,m.attributes[3])
  aG35 = d.averageGain(m3,m.attributes[4])
  aG36 = d.averageGain(m3,m.attributes[5])
  print (aG11,aG12,aG13,aG14,aG15,aG16,'\n',aG21,aG22,aG23,aG24,aG25,aG26,'\n',aG31,aG32,aG33,aG34,aG35,aG36) 
'''
  #sel=d.select(m1, m.attributes[4],0)
  #print(sel)
select_out1=d.select(m1, m.attributes[4], 1)
select_out2=d.select(m1, m.attributes[4], 2)
select_out3=d.select(m1, m.attributes[4], 3)
select_out4=d.select(m1, m.attributes[4], 4)
  #print(select_out1, select_out2, select_out3, select_out4)
'''
  agselect11 = d.averageGain(select_out1,m.attributes[0])
  print(agselect11)
  agselect12 = d.averageGain(select_out1,m.attributes[1])
  print(agselect12)
  agselect13 = d.averageGain(select_out1,m.attributes[2])
  print(agselect13)
  agselect14 = d.averageGain(select_out1,m.attributes[3])
  print(agselect14)
  agselect16 = d.averageGain(select_out1,m.attributes[5])
  print(agselect16)
  agselect21 = d.averageGain(select_out2,m.attributes[0])
  print(agselect21)
  agselect22 = d.averageGain(select_out2,m.attributes[1])
  print(agselect22)
  #agselect = d.averageGain(select_out2,m.attributes[0])
  agselect23 = d.averageGain(select_out2,m.attributes[2])
  print(agselect23)
  agselect24 = d.averageGain(select_out2,m.attributes[3])
  print(agselect24)
  agselect26 = d.averageGain(select_out2,m.attributes[5])
  print(agselect26)
  agselect31 = d.averageGain(select_out3,m.attributes[0])
  print(agselect31)
  agselect32 = d.averageGain(select_out3,m.attributes[1])
  print(agselect32)
'''

tree=d.buildTree(m1, m.attributes)
print(d.check(tree, m1test)

#def partition(data, fraction):
#  ldata = list(data)
#  random.shuffle(ldata)
#  breakPoint = int(len(ldata) * fraction)
#  return ldata[:breakPoint], ldata[breakPoint:]
  
#monk1train, monk1val = partition(m1, 0.6)
#if __name__ == '__main__':
#  main()
