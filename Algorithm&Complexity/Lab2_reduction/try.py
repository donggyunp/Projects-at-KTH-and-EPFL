import sys                                                                  
import fileinput
def readinput():
    para = []
    for line in sys.stdin:
        for cha in line:
            para.append(line.rstrip())                                          
    return para

numbers=readinput()
print(numbers)
