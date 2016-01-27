require('.')
<<<<<<< HEAD:test.lua
--tmp = Sat3(true)
tmp =  torch.load('model/sat3_110.bin')
--tmp:testLM()
for i =11,20 do
print('model/sat3_1' .. tostring(i) ..'.bin')
tmp:train(2)
torch.save('model/sat3_1' .. tostring(i) ..'.bin',tmp)

--tmp = torch.load('model/sat3_01.bin')
=======
tmp = Trans2(true)
--tmp:testLM()
for i= 1,10 do
tmp:train(1)
>>>>>>> 641947e873a541d2b37f6a201252900e784ea26e:run.lua
tmp:evaluate('dev')
end
