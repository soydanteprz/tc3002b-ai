import java.io.*;
import java.util.*;
import java.lang.*;

public class B{
    public static PrintWriter out = new PrintWriter(new BufferedOutputStream(System.out));
    static long MOD = (long) (1e9 + 7);
    //    static long MOD = 998244353;
    static long MOD2 = MOD * MOD;
    static FastReader sc = new FastReader();
    static int pInf = Integer.MAX_VALUE;
    static int nInf = Integer.MIN_VALUE;
    static long ded = (long)(1e17)+9;
    public static void main(String[] args) throws Exception {
        int test = 1;
        test = sc.nextInt();
        for (int i = 1; i <= test; i++){
//            out.print("Case #"+i+": ");
            solve();
        }
        out.flush();
        out.close();
    }
    static void solve(){
        int n = sc.nextInt();
        int m = sc.nextInt();
        int diff = sc.nextInt();
        ArrayList<Pair> A = new ArrayList<>();
        for(int i=  0; i < n; i++){
            int x = sc.nextInt();
            A.add(new Pair(x,i));
        }
        int[] sum = new int[m];
        int[] ans = new int[n];
        Collections.sort(A);
        for(int i=  0; i < n; i++){
            int idx = i%m;
            sum[idx] += A.get(i).x;
            ans[A.get(i).y] = idx+1;
        }
        Arrays.sort(sum);
        if(Math.abs(sum[0]-sum[sum.length-1])>diff){
            out.println("NO");
            return;
        }
        out.println("YES");
        for(int i = 0; i < n; i++){
            out.print(ans[i]+" ");
        }
        out.println();
    }
    static class Pair implements Comparable<Pair> {
        int x;
        int y;
        public Pair(int x, int y) {
            this.x = x;
            this.y = y;
        }
        

        

        
    }
    
    public static long add(long a, long b) {
        return ((a % MOD) + (b % MOD)) % MOD;
    }
    
    //Brian Kernighans Algorithm
    
    //Euclidean Algorithm
    
    //Modular Exponentiation
    
    //AKS Algorithm
    
    
    


    
}
