require('.')
function init_lstm(md)	--初始化lstm单元的参数
	local m1 = md.recurrentModule
	local m2 = m1:findModules('nn.Linear')
	for i,v in pairs(m2) do
		v['weight']= v['weight']:zero() + 100
		v['bias']= v['bias']:zero() + 100
	end
	local m3 = m1:findModules('nn.LinearNoBias')
	for i,v in pairs(m3) do
		v['weight']= v['weight']:zero() + 200
	end
	local m4 = m1:findModules('nn.CMul')
	for i,v in pairs(m4) do
		v['weight']= v['weight']:zero() + 300
	end
end

l1 = nn.LSTM(3,3)
l2 = nn.LSTM(3,3)

inp = {}
for i =1 ,3 do
	inp[i]=torch.randn(3)
end
l2.recurrentModule:share(l1.recurrentModule,'weight','bias')
m3 = l1.recurrentModule:findModules('nn.Linear')[1]
m4 = l2.recurrentModule:findModules('nn.Linear')[1]
s1 = nn.BiSequencer(l1)
s2 = nn.BiSequencer(l2)
init_lstm(l1)
m4:share(m3,'weight','bias')
m3['weight'][2] = 30000
print(l2:getParameters():resize(31,6))
print(l1:getParameters():resize(31,6))
print(s1:getParameters():resize(62,6))
print(s2:getParameters():resize(62,6))
