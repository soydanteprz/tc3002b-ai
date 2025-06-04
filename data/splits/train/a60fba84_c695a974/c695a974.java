import java.io.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


public class Solution {
    static PrintWriter fop = new PrintWriter(System.out);


    public static void main(String[] args) {
        FastScanner fsca = new FastScanner();
        int i, j = 0;
        int t = fsca.nextInt();
        u: while (t-- > 0) {
            int n = fsca.nextInt();
            int a[] = new int[n];
            for(i=0;i<n;i++){
                a[i] = fsca.nextInt();
            }
            int y = Integer.MAX_VALUE;
            long sum[] = new long[n];
            long min1[] = new long[n];
            sum[0] = a[0];
            sum[1] = a[1];
            min1[0] = a[0];
            min1[1] = a[1];
            for(i=2;i<n;i++){
                sum[i] = sum[i-2] + a[i];
                min1[i] = Math.min(min1[i-2],a[i]);
            }
            long val = sum[0]*n + sum[1]*n;
            long min = sum[0]*n + sum[1]*n;
            for(i=2;i<n;i++){
                val = sum[i] + min1[i]*(n-(i+2)/2) + sum[i-1] + min1[i-1]*(n-(i+1)/2);
                min = Math.min(min,val);
            }
            fop.println(min);
        }




















        fop.flush();
        fop.close();


    }
    /*-----------------------------------------------------------------------------------------------------------------------------------------------*/
    // template starts from here :

    

    


    static final Random random = new Random();

    

    


    

    

    
}
