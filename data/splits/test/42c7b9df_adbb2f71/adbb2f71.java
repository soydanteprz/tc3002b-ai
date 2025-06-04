import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;
import java.util.*;

public class cp {
    public static void main(String args[]) throws Exception {
        FastScanner sc = new FastScanner();
        int T = 1;
        T = sc.nextInt();
        PrintWriter pw = new PrintWriter(System.out);
        StringBuilder sb = new StringBuilder();
        while (T-- > 0) {
            solve(sc, pw, sb);
        }
        pw.print(sb);
        pw.close();
    }
    public static void solve(FastScanner sc, PrintWriter pw, StringBuilder sb) throws Exception {
       int n=sc.nextInt();
       int[] h=new int[n];
       for(int i=0;i<n;i++){
            h[i]=sc.nextInt();
       }
       long low=0,high=(long)1e12,ans=low;
       while(low<=high){
            long mid=low+(high-low)/2L;
            if(ok(mid,h)){
                ans=mid;
                low=mid+1;
            }
            else{
                high=mid-1;
            }
       }
       sb.append(ans+"\n");
    }

    public static boolean ok(long value,int[] h){
        int n=h.length;
        int[] copy=new int[n];
        for(int i=0;i<n;i++){
            copy[i]=h[i];
        }
        for(int i=n-1;i-2>=0;i--){
            if(copy[i]<value) return false;
            // we take this condition as the max stones that we can move is the number present in the heap not the copy by the condition di<=hi/3
            long toSub=Math.min(copy[i]-value,h[i])/3L;
            // System.out.println("i: "+i+" toSub:"+toSub);
            copy[i-1]+=toSub;
            copy[i-2]+=toSub*2L;
        }
        // System.out.println(Arrays.toString(copy));
        if(copy[0]<value || copy[1]<value) return false;
        // System.out.println("YES");
        return true;
    }

    

    
    
    
}
