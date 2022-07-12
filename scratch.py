x = 20000 
y = x * 1.025
z = (y - x) / 10

trades = [x]
for i in range(1, 11):
    price = x + (i * z)
    trades.append(price)

    print(sum(trades) / len(trades))

mu = sum(trades) / len(trades)
print(trades[0], trades[-1])
print(((y - mu) / mu))

