




module maxlikeTest

	using FactCheck, maxlike

	facts("Test Data Construction") do

		d = makeData(18)
		@fact d["n"] --> 18

		@fact mean(d["y"]) < 1 --> true
	end

	facts("Test Return value of likelihood") do

		# likelihood should return a real number

	end

	facts("Test return value of gradient") do
		# gradient should not return anything,
		# but modify a vector in place.
		d = makeData()
		gradvec = rand(length(d["beta"]))
		testvec = deepcopy(gradvec)
		r = maxlike.grad(d["beta"],gradvec,d["X"],d["y"],d["dist"])

		@fact r --> nothing 

		@fact gradvec --> not(testvec)

	end

	FactCheck.exitstatus()

end