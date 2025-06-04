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
    static int visited[];
    static HashMap<Pair,Integer> hm;
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
            int n =  scan.nextInt();
            List<List<Integer>> a = new ArrayList<List<Integer>>();
            for(int i=0;i<=n;i++){
                a.add(new ArrayList<Integer>());
            }
            Pair edge[] = new Pair[n-1];
            for(int i=0;i<n-1;i++){
                int x = scan.nextInt();
                int y = scan.nextInt();
                edge[i] = new Pair(x,y);
                a.get(x).add(y);
                a.get(y).add(x);
            }
            int flag=0;
            int start = -1;
            for(int i=1;i<=n;i++){
                if(a.get(i).size()>2)
                flag = 1;
                if(a.get(i).size()==1)
                start = i;
            }
            if(flag==1)
            pw.println(-1);
            else{
                visited = new int[n+1];
                 hm = new HashMap<Pair,Integer>();
                dfs(a,start,2);
                for(int i=0;i<n-1;i++){
                    int x = edge[i].x;
                    int y = edge[i].y;
                   pw.print(hm.get(new Pair(x,y))+" ");
                }
                pw.println();
            }
            pw.flush();
        }
           
             
    }
    static void dfs(List<List<Integer>> a,int start,int parent){
        if(visited[start]==0){
            visited[start] = 1;
        List<Integer> temp = a.get(start);
        int len = temp.size();
        for(int i=0;i<len;i++){
            int end = temp.get(i);
            if(parent==2){
             hm.put(new Pair(start,end),3);
             hm.put(new Pair(end,start),3);
            dfs(a,end,3);
        }
            else{
            hm.put(new Pair(start,end),2);
            hm.put(new Pair(end,start),2);
            dfs(a,end,2);
            }
        }
    }
    }
     
        //static long bin_exp_mod(long a,long n){
            
        //    long res = 1;
            
        //    if(a==0)
        //    return 0;
        //    while(n!=0){
        //        if(n%2==1){
        //            res = ((res)*(a));
        //        }
        //           n = n/2;
        //            a = ((a)*(a));
        //    }
           
        //    return res;
        //}
    //static long bin_exp_mod(long a,long n){
	//    long mod = 1000000007;
	//    long res = 1;
	//    a = a%mod;
	//    if(a==0)
	//    return 0;
	//    while(n!=0){
	//        if(n%2==1){
	//            res = ((res%mod)*(a%mod))%mod;
	//        }
	//           n = n/2;
	//            a = ((a%mod)*(a%mod))%mod;
	//    }
	//    res = res%mod;
	//    return res;
	//}
//   static long gcd(long a,long b){
//       if(a==0)
//       return b;
//       return gcd(b%a,a);
//   }
//   static long lcm(long a,long b){
//       return (a/gcd(a,b))*b;
//   }
}
class Pair{
    Integer x,y;
    Pair(int x,int y){
        this.x = x;
        this.y = y;
    }

    

    
}
