


context("basics") do

	facts("Test Data Construction") do

		d = makeData(18)
		@fact d["n"] --> 18

		@fact mean(d["y"]) < 1 --> true
	end

	facts("Test Return value of likelihood") do

		d = makeData(18)
		@fact isa(maxlike.loglik(rand(3),d) , Real) --> true

	end

	facts("Test return value of gradient") do
		# gradient should not return anything,
		# but modify a vector in place.
		d = makeData()
		gradvec = rand(length(d["beta"]))
		testvec = deepcopy(gradvec)
		r = maxlike.grad!(d["beta"],gradvec,d)

		@fact r --> nothing 

		@fact gradvec --> not(testvec)

	end

	facts("gradient vs finite difference") do
		# gradient should not return anything,
		# but modify a vector in place.
		d = makeData()
		gradvec = ones(length(d["beta"]))
		r = maxlike.grad!(d["beta"],gradvec,d)

		fd = maxlike.test_finite_diff(x->maxlike.loglik(x,d),gradvec,d["beta"],1e-4)

		@fact r --> nothing 

		@fact fd --> true

	end
end

context("test maximization results") do

	facts("maximize returns approximate result") do
		m = maxlike.maximize_like();
		d = makeData();
		@fact m.minimum --> roughly(d["beta"],atol=1e-1)
	end

	facts("maximize_grad returns accurate result") do
		m = maxlike.maximize_like_grad();
		d = makeData();
		@fact m.minimum --> roughly(d["beta"],atol=1e-1)
	end

	facts("gradient is close to zero at max like estimate") do
		m = maxlike.maximize_like_grad();
		d = makeData()
		gradvec = ones(length(d["beta"]))
		r = maxlike.grad!(m.minimum,gradvec,d)

		@fact r --> nothing 

		@fact gradvec --> roughly(zeros(length(gradvec)),atol=1e-5) 

	end

end

context("test against GLM") do
	d = makeData();
	df = hcat(DataFrame(y=d["y"]),convert(DataFrame,d["X"]))
	gg = glm(y~x2+x3,df,Binomial(),ProbitLink())  # don't include intercept column
	m = maxlike.maximize_like_grad_se();

	facts("estimates vs GLM") do

		@fact m[:Estimate] --> roughly(coef(gg),atol=1e-5)

	end

	# # facts("standard errors vs GLM") do

	# # 	# @pending m[:StandardError] --> roughly(stderr(gg),atol=1e-5)

	# end

end

