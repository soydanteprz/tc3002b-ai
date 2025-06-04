import java.io.*;
import java.util.*;
public class A {
    static FastScanner fs;
    public static void main(String[] args) {
        fs=new FastScanner();
        int t = fs.nextInt();
        while (t-->0)
            solve();
    }
    public static void solve() {
        int n = fs.nextInt();
        int[] c = fs.readArray(n);
        long sumEven = c[0];
        long minEven = c[0];
        long sumOdd = c[1];
        long minOdd = c[1];
        long ans = n*minOdd+n*minEven;
        for (int i=2; i<n; i++) {
            if (i%2==0) {
                sumEven+=c[i];
                minEven = Math.min(minEven, c[i]);
            }
            else {
                sumOdd+=c[i];
                minOdd = Math.min(minOdd, c[i]);
            }
            ans = Math.min(ans, sumEven+(n-i/2-1)*minEven+sumOdd+(n-(i+1)/2)*minOdd);
        }
        System.out.println(ans);
    }
    
    static final Random random =new Random();
    

    
}
