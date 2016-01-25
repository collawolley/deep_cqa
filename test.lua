require('.')
tmp = Sat3(true)

--tmp:testLM()
tmp:train(1)
torch.save('model/sat3_01.bin',tmp)
tmp:evaluate('dev')
