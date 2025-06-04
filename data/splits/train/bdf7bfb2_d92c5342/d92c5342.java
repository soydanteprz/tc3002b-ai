import java.io.BufferedReader;
    import java.io.IOException;
    import java.io.InputStreamReader;
    import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.*;
public class Main{
    
    
    static int MOD=1000003;
    public static void main(String[] args) throws IOException {
        FastScanner in = new FastScanner();
        FastWriter out = new FastWriter();
        int t=in.nextInt();
        //int t=1;
        while (t-->0){
            int n=in.nextInt();
            String[] ar=new String[n];
            for (int i = 0; i < n; i++) {
                ar[i]=in.next();
            }
            int ans=0;
            for(char ch='a';ch<='e';ch++){
                int[] res=new int[n];
                for (int i = 0; i < n; i++) {
                    String ss=ar[i];
                    for (int j = 0; j < ss.length(); j++) {
                        if(ss.charAt(j)==ch){
                            res[i]++;
                        }else {
                            res[i]--;
                        }
                    }
                }
                Arrays.sort(res);int max=0,nn=0;
                for (int i = n-1; i>=0; i--) {
                    max+=res[i];
                    if(max>0){
                        nn++;
                    }else {
                        break;
                    }
                }
                ans=Math.max(ans,nn);
            }
            out.println(ans);
        }
        out.close();
    }
    //Arrays.sort(a, (o1, o2) -> (o1[0] - o2[0]));
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}
