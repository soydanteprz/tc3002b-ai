import java.io.*;
import java.util.*;
public class Main {
    static ArrayList<Integer> one=new ArrayList<>();
    static ArrayList<Integer> zero=new ArrayList<>();
    static long dp[][]= new long[5001][5001];
    static long solve(int i,int j){
        if (i==one.size())return 0;
        if (j==zero.size())return Integer.MAX_VALUE;
        if (dp[i][j]!=-1){
           return dp[i][j];
        }
        return dp[i][j]=Math.min(solve(i+1,j+1)+Math.abs(one.get(i)-zero.get(j)),solve(i,j+1));
    }

    public static void main(String[] args) {
        FastScanner sc = new FastScanner();
//        int t=sc.nextInt();
//        while (t-->=1){
            int n=sc.nextInt();
            int a[]=sc.readArray(n);
            for (long i[]:dp){
                Arrays.fill(i,-1);
            }
            for (int i=0;i<n;i++){
                if (a[i]==1)one.add(i);
                else zero.add(i);
            }
            Collections.sort(one);
            Collections.sort(zero);
        System.out.println(solve(0,0));




    }


//        out.flush();



    //------------------------------------if-------------------------------------------------------------------------------------------------------------------------------------------------
    //sieve
    
    static void sort(int[] a) {
        ArrayList<Integer> l = new ArrayList<>();
        for (int i : a) l.add(i);
        Collections.sort(l);
        for (int i = 0; i < a.length; i++) a[i] = l.get(i);
    }
    
    
    
    
    static long mod =(long)(1e9+7);
    static long mod(long x) {
        return ((x % mod + mod) % mod);
    }
    
    static long add(long x, long y) {
        return mod(mod(x) + mod(y));
    }
    
    //Fast Power(logN)
    
    
    
    
    
    
    
    
    
    
}
