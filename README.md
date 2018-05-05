# cs475_project5

# FORWARD PASS
1. start at f1, and do np.dot(f1, fn+1) = mu_1,2 -> store this
2. do np.multiply(mu_1,2 , f2) = p_2
3. do np.dot(p_2, fn+2) = mu_2,3 -> store this
4. do np.multiply(mu_2,3 , f4) = p_3
5. repeat steps 2 and three until we get to np.dot(p_n-1, f2n-1) -> store this
6. get p(xn) by doing np.multiply(mu_n-1,n , fn) because this is a special case -> store this

## BACKWARD PASS
stores all mus from forward pass, each mu is a column vector so we need to transpose it
1. start at fn, and do np.dot(f2n-1, fn.transpose) = mu_n,n-1 -> store transpose of this
2. do np.multiply(fn-1, mu_n,n-1.transpose) = p_n-1' (is a row vector so its easier to code)
3. do np.dot(f2n-2, p_n-1'.transpose) = mu_n-1,n-2 -> store transpose of this
4. do np.multiply(fn-2, mu_n-1,n-2.transpose) = p_n-2'
5. repeat steps 2 and three until we get to np.dot(fn+1 ,p_2'.transpose) -> store this
6. get p(x1) by doing np.multiply(f1 , mu_2,1.transpose) because this is a special case -> store transpose of this

## NOTES
For forward and backwards, you need a base case outside of your for loop