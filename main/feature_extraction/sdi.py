def SDI(x):
    y=x
    N=len(x)
    x=abs(x) 
    L=10 
    for k in range(L-1):    
        j=0
        for i in range(0,len(x)-1,2):
            j=j+1;    
            x[j]=(x[i]+x[i+1])/2     
            y[j]=(y[i]-y[i+1])/2
        x=x[1:round(len(x)/2)]
        y=y[1:round(len(y)/2)]     
    a=x
    s=y
    aa=(a+s)/2
    ss=(a-s)/2
    decomp=math.log10((N/L)*(a*aa-ss*s))
    return decomp