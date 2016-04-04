

module maxlike

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

    export runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm
	# true coeff vector, number of obs, data matrix X (Nxk), response vector y (binary), and a type of parametric distribution; i.e. the standard normal in our case.
 
	function makeData(n=10000)
	beta = [ 1; 1.5; -0.5 ]
	x = hcat(ones(n), randn(n,2))
# X is a nx3 matrix
	d = Normal()
	ys = x*beta + rand(d,n)
	y = map(ys) do z
		if z <= 0
		return 0.0
		else
		return 1.0
		end
	end
	return Dict("beta"=>beta, "numobs"=> n, "X" => x,"y"=> y,"norm" => d)

	end

Data=makeData()
	# log likelihood function at x
	# function loglik(betas::Vector,d::Dict) 
	function loglik(betas::Vector,d=Data)
                  
cdf1(betas) = cdf(d["norm"],betas'*d["X"]')
part1(betas) = d["y"]'*log(cdf1(betas))'
part2(betas) = (1-d["y"])'*log((1-cdf1(betas)))'
return l = (part1(betas)+part2(betas))[1]
		
	end

	# gradient of the likelihood at x
	function grad!(betas::Vector,storage::Vector,d=Data)

#%%# cdf1(betas) = cdf(d["norm"],betas'*d["X"]')
#%%# pdf1(betas) = pdf(d["norm"],betas'*d["X"]')
gradient = zeros(length(d["y"]),3)
for j in 1:3
for i in 1:length(d["y"])
gradient[i,j] = d["X"][i,j]*((d["y"][i]*(pdf(d["norm"],d["X"][i,:]*betas)[1]/cdf(d["norm"],d["X"][i,:]*betas)[1])) - ((1- d["y"][i])*(pdf(d["norm"],d["X"][i,:]*betas)[1]/(1-cdf(d["norm"],d["X"][i,:]*betas)[1]))))
#this is computing each point of the sum (for all i, which are the values of y)
#%%# s = hat(zeros(length(d["y"])), ones(length(d["y"]),2))
#%%# for i in 1:length(d["y"])
#%%# t[i] = (pdf(d["norm"],d["X"][i,:]*betas)/cdf(d["norm"],d["X"][i,:]*betas))[1]
#%%# end
end
storage[j] = sum([gradient[i,j] for i in 1:length(d["y"])])
# this is summing over all the line of the gradient matrix, to compute the value of the gradient for the j-th component.
end
return nothing
	end

	# hessian of the likelihood at x
	function hessian!(betas::Vector,storage::Matrix,d)

#hess = zeros(3, 3, length(d["y"]))
#for j in 1:3
#for k in 1:3
#for i in 1:length(d["y"])
# the hessian formula spread on different lines (because on one line it is less clear)
# I had to use a weird technic (using "[1]" everywhere) to convert the 1-element-Array into a scalar. 
#hess[j, k, i] = (pdf(d["norm"],d["X"][i,:]*betas)[1]
#	*
#	(
#	(d["y"][i]*
#		((pdf(d["norm"],d["X"][i,:]*betas)+ (d["X"][i,:]*betas)[1]*cdf(d["norm"],d["X"][i,:]*betas))[1]/(cdf(d["norm"],d["X"][i,:]*betas)[1]^2)))
#	+
#	(1-d["y"][i])*
#		((pdf(d["norm"],d["X"][i,:]*betas)+ (d["X"][i,:]*betas)[1]*(1-cdf(d["norm"],d["X"][i,:]*betas)[1]))[1]/((1 - cdf(d["norm"],d["X"][i,:]*betas)[1])^2)) 
#	))*(d["X"][i,:]'*d["X"][i,:])[j,k]
#end
#storage[j,k] = - sum([hess[j,k, i] for i in 1:length(d["y"])])
#end
#end
#return nothing

xb = d["X"]*betas
pdf1=pdf(d["norm"],xb)
cdf1=cdf(d["norm"],xb)
part1 = xb.*pdf1./cdf1 .+ (pdf1./cdf1).^2
part2 = (pdf1./1 .- cdf1).^2 .- xb.*pdf1./(1.-cdf1)
fill!(storage, 0.0)
hess = zeros(3, 3, length(d["y"]))
for i in 1:length(d["y"]) 
XX = d["X"][i,:]'*d["X"][i,:]
hess[:,:,i] = d["y"][i].*part1[i].*XX + (1-d["y"][i]).*part2[i].*XX
end
storage[:,:] =-  sum([hess[:,:, i] for i in 1:length(d["y"])])
return nothing

	end


	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)
hessian!(betas::Vector,storage::Matrix,d)
return (inv(-storage))
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
f(t) = loglik(t,makeData())
sol = optimize(f,x0,method=:nelder_mead,grtol=1e-20,ftol=1e-20)
return sol
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=:bfgs)
f(t) = loglik(t,makeData())
gradfct(t,s) = grad!(t,s,makeData())
sol = optimize(f, gradfct, x0, method = :gradient_descent,iterations=5000,grtol=1e-20,ftol=1e-20)
return sol
	end

	# function that maximizes the log likelihood with the gradient
	# and hessian with a call to `optimize` and returns the result
	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=:newton)
f(t) = loglik(t,makeData())
gradfct(t,s) = grad!(t,s,makeData())
hessmat(t,s) = hessian!(t,s,makeData())
sol = optimize(f, gradfct, hessmat, x0, method = :newton,iterations=5000, grtol=1e-20,ftol=1e-20)
return sol
	end

	# function that maximizes the log likelihood with the gradient
	# and computes the standard errors for the estimates
	# should return a dataframe with 3 rows
	# first column should be parameter names
	# second column "Estimates"
	# third column "StandardErrors"
	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=:bfgs)
f(t) = loglik(t,makeData())
gradfct(t,s) = grad!(t,s,makeData())
m4 = optimize(f, gradfct, x0, method = :bfgs,iterations=5000)
stde = se(m4.minimum,makeData())
sol = hcat(["beta1","beta2","beta3"],m4.minimum,stde)
return sol
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the true value.
	function plotLike()


loglik(betas::Vector,d::Dict)


fig,axes = subplots(3,1,figsize=(10,5))
m = 1000
for j in 1:3
betas = makeData()["beta"]
ax = axes[j,1]
l = linspace(betas[j]-5,betas[j]+5,m)
y = zeros(m)

for i in 1:m
betas[j] = collect(l)[i]
y[i] = - loglik(betas,makeData())
end
ax[:plot](l,y)

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





