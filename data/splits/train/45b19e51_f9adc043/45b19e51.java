import java.util.*;
import java.io.*;

public class Main {
    static FastScanner sc = new FastScanner(System.in);
    static PrintWriter pw = new PrintWriter(System.out);
    static StringBuilder sb = new StringBuilder();
    static long mod = (long) 1e9 + 7;

    public static void main(String[] args) throws Exception {
        int n = sc.nextInt();
        for(int i = 0; i < n; i++) solve();
        pw.flush();
    }

    public static void solve() {
        int n = sc.nextInt();
        long[] a = sc.nextLongArray(n);
        long left = 1;
        long right = (long)1e9;
        long[] dec = new long[2];
        while(right-left > 1){
            long mid = (right+left)/2L;
            dec[0] = dec[1] = 0;
            boolean ok = true;
            for(int i = n-1; i >= 2; i--){
                long a1 = a[i] + dec[1];
                if(a1 < mid){
                    ok = false;
                    break;
                }
                long d = (Math.min(a[i],a1-mid))/3L;
                dec[1] = dec[0]+d;
                dec[0] = d*2;
                //pw.println(dec[0] + " " + dec[1] + " " + mid + " " + d);
            }
            if(ok && a[0]+dec[0] >= mid && a[1]+dec[1] >= mid){
                left = mid;
            }else{
                right = mid;
            }
            /*
            for(int i = 2; i < n; i++){
                long a1 = a[i-2]-dec[0];
                long a2 = a[i-1]-dec[1];
                long d = Math.max(0,(mid-a1+1)/2L);
                if(d*3 > a[i]){
                    if(i != 0){
                        a[i] = 
                    }
                    ok = false;
                    break;
                }else{
                    dec[0] = dec[1] - d;
                    dec[1] = d*3;
                }
                //pw.println(dec[0] + " " + dec[1] + " " + mid);
            }
            if(ok && a[n-1]-dec[1] >= mid){
                left = mid;
            }else{
                right = mid;
            }
            */
        }
        pw.println(left);
    }

    
}


