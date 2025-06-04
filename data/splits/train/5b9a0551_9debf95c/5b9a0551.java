import java.io.*;
import java.lang.reflect.Array;
import java.util.*;
public class Main {

    public static void main(String[] args) {
        FastScanner in=new FastScanner();
        PrintWriter out=new PrintWriter(System.out);
        int t=in.nextInt();
        while(t-->0)
        solve(in,out);
        out.close();
    }
    static void solve(FastScanner in,PrintWriter out){
        int n=in.nextInt();
        long a[]=new long[n];
        for (int i = 0; i < n; i++) {
            a[i]=in.nextLong();
        }
        long odd=Integer.MAX_VALUE,even=Integer.MAX_VALUE;
        even=a[0];
        long sum=a[0];
        long ans=Long.MAX_VALUE;
        for (int i = 1; i < n; i++) {
            if(i%2==0) {
                ans=Math.min(ans,(n-i/2)*a[i] + odd*(n-i/2) +sum);
                even=Math.min(even,a[i]);
            } else {
                ans=Math.min(ans,(n-i/2)*a[i] + even*(n-i/2-1) +sum);
                odd=Math.min(odd,a[i]);
            }
            sum+=a[i];
        }
        out.println(ans);
    }

    

    

    
}
