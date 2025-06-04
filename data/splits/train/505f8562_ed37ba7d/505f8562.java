import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;
import static java.lang.Math.*;
import static java.util.Arrays.*;
import static java.lang.System.out;
import static java.lang.Long.MAX_VALUE;

public final class Main{

    FastReader in;
    StringBuffer sb;
    public static void main(String[] args) {
        new Main().run();
    }

    void run(){
        in= new FastReader();
        start();
    }

    void start(){
        sb= new StringBuffer();
        for(int t=in.nextInt();t>0;t--)
            solve();
        out.print(sb);
    }

    long MOD= 1000000007;
    void solve(){
        int n= in.nextInt();
        int k= in.nextInt();
        long[] a=longArr(k);
        long[] t=longArr(k);
        long[] ans= new long[n];
        for(int i=0;i<n;i++){
            ans[i]= Integer.MAX_VALUE;
        }
        for(int i=0;i<k;i++){
            ans[(int) (a[i]-1)]= t[i];
        }
        long[] left= new long[n];
        left[0]= ans[0];
        for(int i=1;i<n;i++) {
            left[i]= min(left[i-1]+1,ans[i]);
        }
        long[] right= new long[n];
        right[n-1]= ans[n-1];
        for(int i=n-2;i>=0;i--) {
            right[i]= min(right[i+1]+1,ans[i]);
        }
        for(int i=0;i<n;i++)
            sb.append(min(left[i], right[i])).append(" ");

        sb.append("\n");
    }

    

    

    long[] longArr(int n){
        long[] res= new long[n];
        for(int i=0;i<n;i++){
            res[i]= in.nextLong();
        }
        return res;
    }

    long MAX= MAX_VALUE;
    int[] precomp= new int[(int) (MAX+1)];
    

    

    

    

    
    

    

    

    
}
