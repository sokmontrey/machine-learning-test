import numpy as np

offset = 892

result = np.genfromtxt('./result.txt', delimiter='\n', dtype=int)

final_str = "PassengerId,Survived\n"
for v in result:
    final_str += str(offset) + "," + str(v) + "\n"
    offset += 1

with open("submission.csv", "w") as f:
    f.write(final_str)
    f.close()
