type_list = [20,10,10,5,5]
result = []
for i in range(len(type_list)):
    x = type_list[i]
    for j in range(x):
        result.append(i)
print(result)