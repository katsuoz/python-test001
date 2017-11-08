# -*- coding: utf-8 -*-

test_dict_1 = {'YEAR':'200', 'MONTH':'1', 'DAY':20}
print(test_dict_1)
del test_dict_1['YEAR']
print(test_dict_1)

print('------------------------------')
print(test_dict_1.keys())
print(test_dict_1.values())
print('------------------------------')

for key, value in test_dict_1.items():
    print(key,value)
