require('.')
tmp = InsSet()
for i = 1,6000 do
	--if (tmp:getNextPair()==nil) then
	--	print(i)
	--	return
	--end
end
for i = 1,10 do
--	print('answer',tmp:getAnswer(i))
end
for i = 1,25000 do
	print('next answer',tmp:getNextAnswer())
end
for i = 1,10 do
--	print('reset answer',tmp:getNextAnswer(true))
end
for i = 1,10 do
	--print('dev',tmp:getNextDev(i))
end
for i = 1,10 do
--	print('test',tmp:getNextTest(i))
end


