require('.')
a =nn.Linear(5,5)
a['weight']:zero()
a['bias']:zero()
for i =1,5 do
	a['weight'][i][i] =1
end
print(a['weight'])
b = torch.Tensor(5):fill(3)
print(b)
print(a:forward(b))
