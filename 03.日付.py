# -*- coding: utf-8 -*-

import datetime

today       = datetime.date.today()
todaydetail = datetime.datetime.today()


print('------------------')
print(today)
print(todaydetail)
print('------------------')

print('------------------')
print(today.year)
print(today.month)
print(today.day)
print('------------------')

print('------------------')
print(today.isoformat())
print(todaydetail.strftime("%Y/%m/%d %H:%M:%S"))


