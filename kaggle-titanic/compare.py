import numpy as np

submission = np.genfromtxt('./submission.csv', delimiter=',', skip_header=1, dtype=int)
gener_sub = np.genfromtxt('./dataset/gender)

print(submission[:, 0])

