import java.util.*;
public class Codeforces {


	// CHECK CONSTRAINTS ALWAYS EDGE CASE MISSED 
	
static int mod = 1000000007;	




public static void main(String[] args) {
     Scanner sc = new Scanner(System.in);
        while(sc.hasNext()) {
   int t   =  sc.nextInt();
   while(t-->0) {
	  	 
	int n =  sc.nextInt();
	long k = sc.nextLong();
	long a[] = new long[n+1];
	for(int i = 1 ; i<=n;i++) {
		a[i] = sc.nextLong();
	}
	a[0] = -1;
	Arrays.sort(a);
	
	long sum = 0;
	for(int i = 1 ; i<=n;i++) {
		sum+=a[i];
	}
	
	if(sum<=k) {
		System.out.println(0);
	}
	else if(sum == k+1) {
		System.out.println(1);
	}
	else {
		long ans = sum-k;
		
		long diff = sum - k;
	         sum = 0;
	  for(int i = n ; i>1 ; i--) {
		long x = 0;
		  sum+=a[i];
		  if(sum - (n-i+1)*a[1] >= diff) {
			  x = 0;
		  }
		  else {
			  x = (long) Math.max(Math.ceil((double)(diff - sum + ((n-i+1)*a[1]))/(n-i+2)) , 0);
	      
		  }
		
		  ans = Math.min(ans, x +n-i+1);
		  
	  }
	  System.out.println(ans);
	         
	         
	}
	
	
	
	

		
	 
	   
	   
	   
	   
	   
	   
	   
   }
     }
    }
    }
