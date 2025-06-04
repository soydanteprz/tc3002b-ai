import java.io.*;
import java.util.*;


public class Main {
    static int i, j, k, n, m, t, y, x, sum = 0;
    static long mod = 998244353;
    static FastScanner fs = new FastScanner();
    static PrintWriter out = new PrintWriter(System.out);
    static String str;
    static long ans;

    public static void main(String[] args) {

        t = fs.nextInt();

        while (t-- > 0) {

            n = fs.nextInt();

            int [] k = fs.readArray(n);

            int [] h = fs.readArray(n);

            List<Pair> arr = new ArrayList<>();

            for(i=0;i<n;i++){
                int a = k[i]-h[i]+1;

                arr.add(new Pair(a,k[i]));
            }

            Collections.sort(arr);

            ans=0;

            List<Pair> comp = new ArrayList<>();
            int temp = arr.get(0).x;
            int tempy = arr.get(0).y;
            for(i=1;i<n;i++){

                if(arr.get(i).x > tempy){
                    comp.add(new Pair(temp, tempy));
                    temp = arr.get(i).x;
                    tempy = arr.get(i).y;
                }

                else tempy = Math.max(tempy, arr.get(i).y);
            }

            comp.add(new Pair(temp, tempy));


            for(i=0;i<comp.size();i++){

                long a = comp.get(i).y - comp.get(i).x +1 ;

                ans+=(((a)*(a+1))/2);
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


    

    

    

    static long add(long a, long b) {
        return ((a % mod) + (b % mod)) % mod;
    }

    

    

    

    


    static class Pair implements Comparable<Pair> {
        public int x, y;

        Pair(int x, int y) {
            this.x = x;
            this.y = y;
        }

        


    }

}
