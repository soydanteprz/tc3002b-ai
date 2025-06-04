import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;

public class D {
    
    public static void main(String[] args)
    {
        FastScanner sc=new FastScanner();
        int t=sc.nextInt();
        PrintWriter pw=new PrintWriter(System.out);
        while(t-->0) {
            int n=sc.nextInt();
            int[] a=sc.readArray(n);
            char[] s=sc.next().toCharArray();
            boolean ok=true;
            ArrayList<Integer> blues=new ArrayList<>();
            ArrayList<Integer> reds=new ArrayList<>();
            for(int i=0;i<n;i++){
                if(s[i]=='B'){
                    blues.add(a[i]);
                } else {
                    reds.add(a[i]);
                }
            }
            Collections.sort(blues);
            Collections.sort(reds);
            for(int i=0;i<blues.size();i++){
                if(blues.get(i)<(i+1)){
                    ok=false;
                    break;
                }
            }
            int start=blues.size()+1;
            for(int i=0;i<reds.size();i++){
                if(reds.get(i)>(start++)){
                    ok=false;
                    break;
                }
            }
            if(ok){
               pw.println("YES");
            } else {
                pw.println("NO");
            }
        }
        pw.flush();
    }

    
}
