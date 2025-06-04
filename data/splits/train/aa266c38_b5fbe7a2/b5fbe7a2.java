import java.util.*;
import java.io.*;

public class Codeforces {
    final static long mod = 1000000007;

    public static void main(String[] args) {
        FastReader sc = new FastReader();
        int t = sc.nextInt();
        while(t-->0){
           int n=sc.nextInt();
           int m=sc.nextInt();
           int x=sc.nextInt();
           
           Pair a[]=new Pair[n];
           
           for(int i=0;i<n;i++){
               a[i]=new Pair(sc.nextInt(), i+1);
           }
           
           Arrays.sort(a);
           
           int b[]=new int[n+1];
           int k=m;
           
           for(int i=n-1;i>=0;i--){
               b[a[i].y]=k;
               k--;
               if(k==0)
                k=m;
           }
           
           System.out.println("YES");
           for(int i=1;i<=n;i++){
               System.out.print(b[i]+" ");
           }
           System.out.println();
           
        }
    }
    
    
    

    static class Pair implements Comparable<Pair> {
        public final int x;
        public final int y;

        public Pair(int x, int y) {
            this.x = x;
            this.y = y;
        }

        
        
    }

    

    

    

    

    

    

}
