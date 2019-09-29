import os

from classify.sen_classify import main as classify

os.chdir("./classify")
result = classify([" ", "I", "want", "to", "eaten", "apple", "."])
print(result)