import re

with open('cabt_extracted/cabt.py', 'r') as f:
    content = f.read()

# We can search for the class definitions of XerneasEX and Houndour
x_ex = re.search(r'class XerneasEX.*?def attacks\(self\):.*?(return \[.*?\])', content, re.DOTALL)
if x_ex: print("XerneasEX attacks:", x_ex.group(1))

hound = re.search(r'class Houndour.*?def attacks\(self\):.*?(return \[.*?\])', content, re.DOTALL)
if hound: print("Houndour attacks:", hound.group(1))

