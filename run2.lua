require('.')
tmp = Trans2(true)
--tmp:testLM()
--[
for i= 1,50 do
	print('Training:',i)
	tmp:train(1)
end
tmp:evaluate('dev')
for i= 1,50 do
	print('epoch:',i+50)
	tmp:train(1)
	tmp:evaluate('dev')
end
--]
