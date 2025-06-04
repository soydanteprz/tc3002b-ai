import static java.lang.Integer.MAX_VALUE;
import static java.lang.Integer.parseInt;
import static java.lang.Long.parseLong;
import static java.lang.System.exit;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.StringTokenizer;

public class B {

    static void solve() throws Exception {
        int t=scanInt();
        while (t-->0){
            int n=scanInt();
            int []arr=new int[n];
            int low=Integer.MAX_VALUE;
            int high=0;
            for (int i=0;i<n;i++){
                arr[i]=scanInt();
                low=Math.min(low,arr[i]);
                high=Math.max(high,arr[i]);
            }
            int ans=low;
            while (low<=high){
                int mid=low+(high-low)/2;
                boolean flag=true;
                int []temp=new int[n];
                for (int i=0;i<n;i++){
                    temp[i]=arr[i];
                }

                for (int i=n-1;i>=2;i--){
                    if(temp[i]<mid){
                        flag=false;
                        break;
                    }

                    int rem=Math.min(temp[i]-mid,arr[i]);
                    int taken=rem/3;
                    temp[i]-=3*taken;
                    temp[i-1]+=taken;
                    temp[i-2]+=2*taken;

                }

                if(temp[0]<mid || temp[1]<mid){
                    flag=false;
                }
                if(flag){
                    low=mid+1;
                    ans=mid;
                }else {
                    high=mid-1;
                }
            }

            System.out.println(ans);
        }

    }

    static int scanInt() throws IOException {
        return parseInt(scanString());
    }

    

    static String scanString() throws IOException {
        while (tok == null || !tok.hasMoreTokens()) {
            tok = new StringTokenizer(in.readLine());
        }
        return tok.nextToken();
    }

    static BufferedReader in;
    static PrintWriter out;
    static StringTokenizer tok;

    public static void main(String[] args) {
        try {
            in = new BufferedReader(new InputStreamReader(System.in));
            out = new PrintWriter(System.out);
            solve();
            in.close();
            out.close();
        } catch (Throwable e) {
            e.printStackTrace();
            exit(1);
        }
    }
}
