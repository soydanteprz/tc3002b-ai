import java.util.*;
import java.io.*;
public class Solution {
    
    
    
    
    

    static int mod = (int)1e9+7;
    static int INF = Integer.MAX_VALUE;
    static PrintWriter out;
    static FastReader sc ;
    public static void main(String[] args) throws IOException {
        sc = new FastReader();
        out = new PrintWriter(System.out);
        // primes();
        // ================================ //
        int test = sc.nextInt();
        while (test-- > 0) {
            int n = sc.nextInt();
            String s = sc.nextLine();
            String t = sc.nextLine();
            solver(s,t, n);
        }
        // ================================ //
        // int n = sc.nextInt();
        // solver();
        // ================================ //
        out.flush();
    }

    public static void solver(String s, String t, int n) {
        int diff = 0;
        int one = 0;
        int zero = 0;
        for(int i=0;i<n;i++){
            if(s.charAt(i)!=t.charAt(i)){
                diff++;
                if(s.charAt(i)=='1')one++;
            }
            else{
                if(s.charAt(i)=='1')zero++;
            }
        }
        if(diff==0){
            out.println(0);
            return;
        }
        int res = INF;
        if(diff%2==0 && one==(diff-one)){
            res = diff;
        }
        if(n-diff-1>=0 && (n-diff-1)%2==0 && zero>0 && (n-diff-zero)==zero-1){
            res = Math.min(res, (n-diff-1)+1);
        }
        out.println(res==INF?-1:res);
    }
}
