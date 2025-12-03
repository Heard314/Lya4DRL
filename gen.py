# import random
# import math

# def generate_points(num_points, min_radius, max_radius):
#     points = []

#     while len(points) < num_points:
#         x = random.uniform(-max_radius, max_radius)
#         y = random.uniform(-max_radius, max_radius)
#         distance = math.sqrt(x**2 + y**2)
        
#         if min_radius <= distance <= max_radius:
#             points.append((x, y))

#     return points

# # 生成50个坐标点，满足 100 <= sqrt(x^2 + y^2) <= 500
# points = generate_points(num_points=50, min_radius=100, max_radius=500)

# # 打印结果
# print("[",end="")
# for i, (x, y) in enumerate(points, 1):
#     print(x,end=",")
# print("]",end="")

# print("[",end="")
# for i, (x, y) in enumerate(points, 1):
#     print(y,end=",")
# print("]",end="")
print(16.5*1024*1024/3/640/640/8)
print((3.0*224*224*8)/(1024.0*1024))