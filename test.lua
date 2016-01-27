require('.')
--tmp = Sat3(true)
tmp =  torch.load('model/sat3_110.bin')
--tmp:testLM()
for i =11,20 do
print('model/sat3_1' .. tostring(i) ..'.bin')
tmp:train(2)
torch.save('model/sat3_1' .. tostring(i) ..'.bin',tmp)

--tmp = torch.load('model/sat3_01.bin')
tmp:evaluate('dev')
end
