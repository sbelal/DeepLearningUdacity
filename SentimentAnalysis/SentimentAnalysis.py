age = {"Sal":36, 'Naz':29}

age["Naz"] = 30
age["Baz"] = 32
#del age["Sal"]


print(age["Naz"])
print(age)
print(age["Sal"])
print(age["Baz"])

print(42 in age.values())