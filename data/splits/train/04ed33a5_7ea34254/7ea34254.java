import java.util.*;
import java.lang.*;
import java.io.*;
/* Name of the class has to be "Main" only if the class is public. */
public class Codechef
{ 
//    static int mod = 998244353 ;
//    static int N = 200005;
//    static long factorial_num_inv[] = new long[N+1];
//    static long natual_num_inv[] = new long[N+1];
//    static long fact[] = new long[N+1];
//    static void InverseofNumber() 
//{ 
//    natual_num_inv[0] = 1;
//    natual_num_inv[1] = 1; 
//    for (int i = 2; i <= N; i++) 
//        natual_num_inv[i] = natual_num_inv[mod % i] * (mod - mod / i) % mod; 
//} 
//static void InverseofFactorial() 
//{ 
//    factorial_num_inv[0] = factorial_num_inv[1] = 1; 
  
//    for (int i = 2; i <= N; i++) 
//        factorial_num_inv[i] = (natual_num_inv[i] * factorial_num_inv[i - 1]) % mod; 
//} 
//static long nCrModP(long N, long R) 
//{ 
//    long ans = ((fact[(int)N] * factorial_num_inv[(int)R]) % mod * factorial_num_inv[(int)(N - R)]) % mod; 
//    return ans%mod; 
//}
    //static boolean prime[];
    //static void sieveOfEratosthenes(int n)
    //{
    //    prime = new boolean[n+1];
        
    //    for (int i = 0; i <= n; i++)
    //        prime[i] = true;
 
    //    for (int p = 2; p * p <= n; p++)
    //    {
    //        // If prime[p] is not changed, then it is a
    //        // prime
    //        if (prime[p] == true)
    //        {
    //            // Update all multiples of p
    //            for (int i = p * p; i <= n; i += p)
    //                prime[i] = false;
    //        }
    //    }
 
        
    //}
    
	public static void main (String[] args) throws java.lang.Exception
	{
    //    InverseofNumber();
    //    InverseofFactorial();
    //   fact[0] = 1; 
    //   for (long i = 1; i <= 2*100000; i++)
    //   { 
    //       fact[(int)i] = (fact[(int)i - 1] * i) % mod; 
    //   }
    
		FastReader scan = new FastReader();
		PrintWriter pw = new PrintWriter(System.out);
        int t = scan.nextInt();
        while(t-->0){
         int n= scan.nextInt();
         long a[] = new long[n];
         for(int i=0;i<n;i++){
             a[i] = scan.nextLong();
         }
         long b[] = new long[n];
         int i=0;
         if(n%2==1){
           if(a[0]+a[1]!=0){
           b[0] = -1*a[2];
           b[1] = -1*a[2];
           b[2] = a[0]+a[1];}
           else if(a[0]+a[2]!=0){
            b[0] = -1*a[1];
            b[2] = -1*a[1];
            b[1] = a[0]+a[2];
           }
           else{
            b[2] = -1*a[0];
            b[1] = -1*a[0];
            b[0] = a[2]+a[1]; 
           }
           i = 3;
         }
         else{
             b[0] = -a[1];
             b[1] = a[0];
             i=2;
         }
         while(i<n-1){
             b[i] = -a[i+1];
             b[i+1] = a[i];
             i = i+2;
         }
         for(int j=0;j<n;j++){
             pw.print(b[j]+" ");
         }
         pw.println();
         pw.flush();
        }
    
    }
    
//   static long gcd(long a,long b){
//       if(a==0)
//       return b;
//       return gcd(b%a,a);
//   }
//   static long lcm(long a,long b){
//       return (a/gcd(a,b))*b;
//   }
}
