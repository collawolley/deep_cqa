require('nn')
a={}
a[1] = torch.ones(5)*1
a[2] = torch.ones(5)*2
d = nn.Dropout()
for i = 1,100 do 
--	print(d:forward(a[1]))
end
a[1][1] = 0
a[2][2] = 0

--c = nn.CosineDistance()
--print(c:forward({a[1],a[2]}))
--e = c:backward({a[1],a[2]},torch.Tensor({0.5}))
--print(e[1],e[2])
--print(a[1]:cmul(a[2]))

err = torch.Tensor({-0.2})
gold = torch.Tensor({-1})
m = nn.MarginCriterion(0.1)
print(m:forward(err,gold))
print(m:backward(err,gold)[1])

