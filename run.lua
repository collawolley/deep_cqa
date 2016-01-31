require('.')
tmp = Sat3(true)
--tmp =  torch.load('model/sat3_120.bin')
--tmp:testLM()
for i =1,50 do
	print('model/sat3_2_' .. tostring(i) ..'.bin')
	tmp:train(1)
	if i %5 ==0 then
		torch.save('model/sat3_2_' .. tostring(i) ..'.bin',tmp)
		tmp:evaluate('dev')
	end
end

