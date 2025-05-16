import time
current_time = time.localtime()
formatted_time = time.strftime('%Y_%m_%d_%H_%M_%S', current_time)
print(f"metrics_{formatted_time}.txt")