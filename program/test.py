
input_list = 'aaabbccccccaa'
i = 0
result = []
while i < len(input_list):
    n = 1
    while input_list[i] == input_list[i+1]:
        n += 1
        i += 1
    result.append(input_list[i])
    result.append(str(n))
    if i == len(input_list)-1:
        break
    else:
        i += 1
print("".join(result))