# import numpy as np
# arr_rand = np.array([8, 8, 3, 7, 7, 0, 4, 2, 5, 2])
# print("Array: ", arr_rand)

# f = open("output1.txt","a")
# f.write("Predicted relation:" + str(np.argmax(arr_rand)) + "\n")
# f.write("Actual relation:" + str(np.argmax(arr_rand) + "\n")

	# Create an array
import numpy as np
arr_rand = np.array([8, 8, 3, 7, 7, 0, 4, 2, 5, 2])
print("Array: ", arr_rand)
f = open("output1.txt","a")
# Positions where value > 5
index_gt5 = np.where(arr_rand > 5)
print("Positions where value > 5: ", index_gt5)


ind = np.unravel_index(np.argmax(arr_rand),arr_rand.shape)
x = arr_rand[ind]
y = arr_rand[ind]
f.write("Predicted relation:" + str(x) + "\n")
f.write("Actual relation:" + str(y) + "\n")