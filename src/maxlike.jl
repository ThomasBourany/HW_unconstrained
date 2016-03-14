

module maxlike

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, PyPlot, DataFrames, Debug

	"""
    `input(prompt::AbstractString="")`
  
    Read a string from STDIN. The trailing newline is stripped.
  
    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    function input(prompt::AbstractString="")
        print(prompt)
        return chomp(readline())
    end

    export maximize_like_grad, runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	function makeData(n=10000)
		beta = [ 1; 1.5; -0.5 ]
		srand(3122015)
		numobs = n
		X = hcat(ones(numobs), randn(numobs,2))    # n,k
		epsilon = randn(numobs)
		Y = X * beta + epsilon
		y = 1.0 * (Y .> 0)
		norm = Normal(0,1)	# create a normal distribution object with mean 0 and variance 1
		return Dict("beta"=>beta,"n"=>numobs,"X"=>X,"y"=>y,"dist"=>norm)
	end


	# log likelihood function at x
	# function loglik(betas::Vector,X::Matrix,y::Vector,distrib::UnivariateDistribution) 
	function loglik(betas::Vector,d::Dict)
		xbeta     = d["X"]*betas	# (n,1)
		G_xbeta   = cdf(d["dist"],xbeta)	# (n,1)
		loglike   = d["y"] .* log(G_xbeta) .+ (1-d["y"]) .* log(1-G_xbeta)  # (n,1)
		objective = -sum(loglike)
		return objective

	end

	# gradient of the likelihood at x
	function grad!(betas::Vector,storage::Vector,d)
		xbeta     = d["X"]*betas	# (n,1)
		G_xbeta   = cdf(d["dist"],xbeta)	# (n,1)
		g_xbeta   = pdf(d["dist"],xbeta)	# (n,1)
		# println("G_xbeta = $G_xbeta")
		# println("g_xbeta = $g_xbeta")
		# storage[:]= -mean((d["y"] .* g_xbeta ./ G_xbeta - (1-d["y"]) .* g_xbeta ./ (1-G_xbeta)) .* d["X"],1)
		storage[:]= -sum((d["y"] .* g_xbeta ./ G_xbeta - (1-d["y"]) .* g_xbeta ./ (1-G_xbeta)) .* d["X"],1)
		return nothing
	end

	# hessian of the likelihood at x
	function hessian!(betas::Vector,storage::Matrix,d)
		xbeta     = d["X"]*betas	# (n,1)
		G_xbeta   = cdf(d["dist"],xbeta)	# (n,1)
		g_xbeta   = pdf(d["dist"],xbeta)	# (n,1)
		xb1 = xbeta .* g_xbeta ./ G_xbeta .+ (g_xbeta ./ G_xbeta).^2
		xb2 = (g_xbeta ./ (1 .- G_xbeta)).^2  .- xbeta .* g_xbeta ./ (1 -G_xbeta) 
		fill!(storage,0.0)
		# storage[:,:] = zeros(length(betas),length(betas))
		
		for i in 1:d["n"]
			XX = d["X"][i,:]' * d["X"][i,:]   #k,k
			storage[:,:] = storage .+ (xb1[i] + xb2[i]) * XX
		end 

		return nothing
	end


	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)
		h = zeros(length(betas),length(betas))
		hessian!(betas,h,d)
		inv(h)
	end

	"""
	standard errors
	"""
	function se(betas::Vector,d::Dict)
		sqrt(diag(inv_observedInfo(betas,d)))
	end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result
	function maximize_like(x0=[0.8,1.0,-0.1],meth=:nelder_mead)
		d = makeData(10000)
		res = optimize(arg->loglik(arg,d),x0, method = meth, iterations = 100)
		return res
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=:bfgs)
		d = makeData(10000)
		# storage = zeros(length(d["beta"]))
		res = optimize((arg)->loglik(arg,d),(arg,g)->grad!(arg,g,d),x0, method = meth, iterations = 1000,grtol=1e-20,ftol=1e-20)
		return res
	end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=:newton)
		d = makeData(10000)
		# storage = zeros(length(d["beta"]))
		res = optimize((arg)->loglik(arg,d),(arg,g)->grad!(arg,g,d),(arg,g)->hessian!(arg,g,d),x0, method = meth, iterations = 1000,grtol=1e-20,ftol=1e-20)
		return res
	end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=:bfgs)
		d = makeData(10000)
		# storage = zeros(length(d["beta"]))
		res = optimize((arg)->loglik(arg,d),(arg,g)->grad!(arg,g,d),x0, method = meth, iterations = 1000,grtol=1e-20,ftol=1e-20)
		ses = se(res.minimum,d)
		return DataFrame(Parameter=["beta$i" for i=1:length(x0)], Estimate=res.minimum, StandardError=ses)
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value.
	function plotLike()
		d = makeData(10000)
		ngrid = 100
		pad = 1
		k = length(d["beta"])
		beta0 = repmat(d["beta"]',ngrid,1)
		values = zeros(ngrid)
		fig,axes = subplots(1,k)
		currplot = 0
		for b in 1:k
			currplot += 1
			xaxis = collect(linspace(d["beta"][b]-pad,d["beta"][b]+pad,ngrid))
			betas = copy(beta0)
			betas[:,b] = xaxis
			for i in 1:ngrid
				values[i] = loglik(betas[i,:][:],d)
			end
			ax = axes[currplot]
			ax[:plot](xaxis,values)
			ax[:set_title]("beta $currplot")
			ax[:axvline](x=d["beta"][b],color="red")

		end
		fig[:canvas][:draw]()
	end

	function runAll()

		plotLike()
		m1 = maximize_like()
		m2 = maximize_like_grad()
		m3 = maximize_like_grad_hess()
		m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like: $(m1.minimum)")
		println("maximize_like_grad: $(m2.minimum)")
		println("maximize_like_grad_hess: $(m3.minimum)")
		println("maximize_like_grad_se: $m4")
		println("")
		println("running tests:")
		include("test/runtests.jl")
		println("")
		ok = input("enter y to close this session.")
		if ok == "y"
			quit()
		end
	end


end





