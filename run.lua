require('.')
--tmp = Sat3(true)
tmp =  torch.load('model/sat3_120.bin')
--tmp:testLM()
for i =21,30 do
	print('model/sat3_1' .. tostring(i) ..'.bin')
	tmp:train(2)
	torch.save('model/sat3_1' .. tostring(i) ..'.bin',tmp)
end

