import java.io.*;
import java.util.*;


public class Main {
    static int i, j, k, n, m, t, y, x, sum = 0;
    static long mod = 998244353;
    static FastScanner fs = new FastScanner();
    static PrintWriter out = new PrintWriter(System.out);
    static String str;

    public static void main(String[] args) throws IOException {

        t = fs.nextInt();

        while (t-- > 0) {

            n = fs.nextInt();
            long k = fs.nextLong();

            long sum=0;

            int[] arr = fs.readArray(n);

            for(i=0;i<n;i++)
                sum+=arr[i];

            Arrays.sort(arr);

            long[] suffix = new long[n];

            suffix[n-1] = arr[n-1]-arr[0];

            for(i=n-2;i>=0;i--){
                suffix[i] = suffix[i+1]+arr[i]-arr[0];
            }

            long ans = sum-k;

            if(ans<=0){
                out.println(0);
                continue;
            }

            for(i=n-1;i>0;i--){

                long temp = sum-k;
                long ansHere;
                if(suffix[i]>=temp){
                    ansHere = n-i;
                }

                else{
                    long y = temp - suffix[i];

                    long x = n-i+1;

                    long moves = y/x;

                    if(y%x!=0)
                        moves++;

                    ansHere = moves+(n-i);
                }

                ans = Math.min(ans,ansHere);
            }

            out.println(ans);

        }

        out.close();
    }

   /*static long nck(int n , int k){
        long a = fact[n];
        long b = modInv(fact[k]);
        b*= modInv(fact[n-k]);
        b%=mod;

        return (a*b)%mod;
    }

   static void populateFact(){
        fact[0]=1;

        fact[1] = 1;

        for(i=2;i<300005;i++){
            fact[i]=i*fact[i-1];
            fact[i]%=mod;
        }
    }
    */


    

    

    

    

    


    

}
